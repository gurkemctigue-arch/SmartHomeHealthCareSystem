"""
多模态智能医疗家庭助手·成果大屏
Flask 主程序
"""
import logging
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, request

from config import Config, setup_logging
from database.db import init_db
from services.video_service import generate_video_stream, get_detect_mode, set_detect_mode
from services.stats_service import (
    get_emotion_summary,
    get_latest_detections,
    get_medicine_stats,
    get_overview_data,
    get_trend_data,
)
from services.alert_service import (
    acknowledge_alert,
    get_latest_alerts,
    resolve_medicine_alerts,
    sync_medicine_alerts,
)
from services.medicine_service import (
    add_medicine,
    delete_medicine,
    get_all_medicines,
    get_medicine_by_id,
    update_medicine,
)
from services.weather_service import (
    WeatherServiceError,
    get_weather_snapshot,
    search_locations,
)
from services.wellness_service import (
    ProfileValidationError,
    get_daily_recommendation,
    get_health_profile,
    save_health_profile,
)
from services.recipe_service import RecipeNotFoundError, get_recipe_detail
from models.llm_client import ask_local_llm
from quest_routes import quest_bp
from vital_twin_routes import vital_twin_bp

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)


def create_app(test_config=None):
    """创建 Flask 应用实例"""
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    app.register_blueprint(quest_bp)
    app.register_blueprint(vital_twin_bp)

    # 初始化数据库
    init_db(app)

    # ── 页面路由 ──────────────────────────────────────────

    @app.route("/")
    def index():
        """返回大屏首页"""
        return render_template("index.html")

    @app.route("/favicon.ico")
    def favicon():
        """Avoid a browser-generated 404 when no custom favicon is configured."""
        return "", 204

    # ── 视频流接口 ────────────────────────────────────────

    @app.route("/video_feed")
    def video_feed():
        """摄像头视频流（MJPEG）"""
        return Response(
            generate_video_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    # ── 数据接口 ──────────────────────────────────────────

    @app.route("/api/overview")
    def overview():
        """顶部概览数据"""
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_overview_data()
        })

    @app.route("/api/stats/medicine")
    def medicine_stats():
        """药品类别统计"""
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_medicine_stats()
        })

    @app.route("/api/stats/trend")
    def trend_stats():
        """近7日检测趋势"""
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_trend_data()
        })

    @app.route("/api/alerts/latest")
    def latest_alerts():
        """最新告警"""
        limit = request.args.get("limit", 5, type=int)
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_latest_alerts(limit=limit)
        })

    @app.route("/api/alerts")
    def alerts():
        """告警列表，支持级别和处置状态筛选。"""
        limit = request.args.get("limit", 50, type=int)
        status = request.args.get("status", "all")
        level = request.args.get("level", "all")
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_latest_alerts(limit=limit, status=status, level=level),
        })

    @app.route("/api/alerts/<int:alert_id>/acknowledge", methods=["POST"])
    def acknowledge_alert_route(alert_id):
        """确认一条告警已经处理。"""
        if not acknowledge_alert(alert_id):
            return jsonify({"code": 404, "message": "告警不存在", "data": None}), 404
        return jsonify({"code": 200, "message": "ok", "data": {"id": alert_id}})

    # ── 药品管理接口 ────────────────────────────────────────

    @app.route("/api/medicines", methods=["GET", "POST"])
    def medicines():
        if request.method == "GET":
            return jsonify({"code": 200, "message": "success", "data": get_all_medicines()})

        payload, error = _validate_medicine_payload(request.get_json(silent=True))
        if error:
            return jsonify({"code": 400, "message": error, "data": None}), 400
        medicine_id = add_medicine(**payload)
        if medicine_id is None:
            return jsonify({"code": 500, "message": "药品保存失败", "data": None}), 500
        medicine = get_medicine_by_id(medicine_id)
        sync_medicine_alerts(medicine)
        return jsonify({"code": 201, "message": "created", "data": medicine}), 201

    @app.route("/api/medicines/<int:medicine_id>", methods=["GET", "PUT", "DELETE"])
    def medicine_detail(medicine_id):
        medicine = get_medicine_by_id(medicine_id)
        if medicine is None:
            return jsonify({"code": 404, "message": "药品不存在", "data": None}), 404

        if request.method == "GET":
            return jsonify({"code": 200, "message": "success", "data": medicine})
        if request.method == "DELETE":
            resolve_medicine_alerts(medicine["name"])
            delete_medicine(medicine_id)
            return jsonify({"code": 200, "message": "deleted", "data": {"id": medicine_id}})

        payload, error = _validate_medicine_payload(request.get_json(silent=True))
        if error:
            return jsonify({"code": 400, "message": error, "data": None}), 400
        if not update_medicine(medicine_id, **payload):
            return jsonify({"code": 404, "message": "药品不存在", "data": None}), 404
        updated = get_medicine_by_id(medicine_id)
        sync_medicine_alerts(updated)
        return jsonify({"code": 200, "message": "updated", "data": updated})

    @app.route("/api/detections/latest")
    def latest_detections():
        limit = request.args.get("limit", 20, type=int)
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_latest_detections(limit=limit),
        })

    @app.route("/api/emotions/summary")
    def emotion_summary():
        days = request.args.get("days", 7, type=int)
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_emotion_summary(days=days),
        })

    # ── 问答接口 ──────────────────────────────────────────

    @app.route("/api/mode", methods=["GET", "POST"])
    def detect_mode():
        """获取或切换检测模式：medicine / emotion / both"""
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            mode = body.get("mode", "both")
            if mode in ("medicine", "emotion", "both"):
                set_detect_mode(mode)
                return jsonify({"code": 200, "message": "ok", "data": {"mode": mode}})
            return jsonify({"code": 400, "message": "invalid mode", "data": None}), 400
        return jsonify({"code": 200, "message": "ok", "data": {"mode": get_detect_mode()}})

    @app.route("/api/chat", methods=["POST"])
    def chat():
        """健康科普问答"""
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()

        if not question:
            return jsonify({"code": 400, "message": "问题不能为空", "data": None}), 400

        answer = ask_local_llm(question)

        # 持久化聊天记录
        try:
            from database.db import get_db
            db = get_db()
            db.execute(
                "INSERT INTO chat_record (question, answer) VALUES (?, ?)",
                (question, answer)
            )
            db.commit()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("聊天记录保存失败: %s", e)

        return jsonify({
            "code": 200,
            "message": "success",
            "data": {
                "question": question,
                "answer": answer
            }
        })

    # 天时智养：实时环境数据、健康档案与每日确定性推荐

    @app.route("/api/wellness/daily")
    def wellness_daily():
        latitude = request.args.get("latitude")
        longitude = request.args.get("longitude")
        preferred_name = request.args.get("location_name", "")
        force = request.args.get("force", "false").lower() in ("1", "true", "yes")
        try:
            weather = get_weather_snapshot(
                latitude,
                longitude,
                preferred_name=preferred_name,
                force=force,
            )
            recommendation = get_daily_recommendation(weather)
        except ValueError as exc:
            return jsonify({"code": 400, "message": str(exc), "data": None}), 400
        except WeatherServiceError as exc:
            logger.warning("Wellness environment request failed: %s", exc)
            return jsonify({"code": 503, "message": str(exc), "data": None}), 503
        return jsonify({
            "code": 200,
            "message": "success",
            "data": {"weather": weather, "recommendation": recommendation},
        })

    @app.route("/api/wellness/locations")
    def wellness_locations():
        try:
            locations = search_locations(request.args.get("q", ""))
        except ValueError as exc:
            return jsonify({"code": 400, "message": str(exc), "data": None}), 400
        except WeatherServiceError as exc:
            return jsonify({"code": 503, "message": str(exc), "data": None}), 503
        return jsonify({"code": 200, "message": "success", "data": locations})

    @app.route("/api/wellness/profile", methods=["GET", "PUT"])
    def wellness_profile():
        if request.method == "GET":
            return jsonify({"code": 200, "message": "success", "data": get_health_profile()})
        try:
            profile = save_health_profile(request.get_json(silent=True))
        except ProfileValidationError as exc:
            return jsonify({"code": 400, "message": str(exc), "data": None}), 400
        return jsonify({"code": 200, "message": "updated", "data": profile})

    @app.route("/api/wellness/recipes/<recipe_id>")
    def wellness_recipe_detail(recipe_id):
        """Return a curated recipe with an explicit current-profile safety check."""
        try:
            recipe = get_recipe_detail(recipe_id, get_health_profile())
        except RecipeNotFoundError as exc:
            return jsonify({"code": 404, "message": str(exc), "data": None}), 404
        return jsonify({"code": 200, "message": "success", "data": recipe})

    # ── 健康检查接口 ──────────────────────────────────────

    @app.route("/api/health")
    def health():
        """服务健康检查"""
        import platform
        status = {
            "service": "多模态智能医疗家庭助手",
            "status": "running",
            "python": platform.python_version(),
        }
        # 检查各组件状态
        try:
            from models.llm_client import is_llm_available, get_llm_backend, get_rag_status
            status["llm_available"] = is_llm_available()
            status["llm_backend"] = get_llm_backend()
            status["rag_status"] = get_rag_status()
        except Exception:
            status["llm_available"] = False
            status["llm_backend"] = "error"

        try:
            from services.video_service import detector as _det
            status["detector"] = "YOLO-OBB" if _det.is_real_model else "MOCK"
        except Exception:
            status["detector"] = "unavailable"

        try:
            from services.video_service import emotion_model as _emo
            status["emotion"] = "CNN" if _emo.is_real_model else "MOCK"
        except Exception:
            status["emotion"] = "unavailable"

        try:
            from database.db import get_db
            db = get_db()
            db.execute("SELECT 1")
            status["database"] = "connected"
        except Exception:
            status["database"] = "disconnected"

        return jsonify({"code": 200, "message": "ok", "data": status})

    # ── 错误处理 ──────────────────────────────────────────

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"code": 404, "message": "接口不存在", "data": None}), 404

    @app.errorhandler(500)
    def server_error(_error):
        return jsonify({"code": 500, "message": "服务器内部错误", "data": None}), 500

    return app


def _validate_medicine_payload(body):
    """Validate and normalize medicine JSON used by create and update routes."""
    if not isinstance(body, dict):
        return None, "请求内容必须是 JSON 对象"

    name = str(body.get("name") or "").strip()
    if not name:
        return None, "药品名称不能为空"
    if len(name) > 80:
        return None, "药品名称不能超过 80 个字符"

    try:
        stock = int(body.get("stock", 0))
    except (TypeError, ValueError):
        return None, "库存必须是整数"
    if stock < 0 or stock > 99999:
        return None, "库存必须在 0 到 99999 之间"

    expire_date = body.get("expire_date") or None
    if expire_date:
        try:
            datetime.strptime(str(expire_date), "%Y-%m-%d")
        except ValueError:
            return None, "有效期格式必须为 YYYY-MM-DD"

    category = str(body.get("category") or "").strip() or None
    description = str(body.get("description") or "").strip() or None
    if category and len(category) > 40:
        return None, "类别不能超过 40 个字符"
    if description and len(description) > 300:
        return None, "备注不能超过 300 个字符"

    return {
        "name": name,
        "category": category,
        "expire_date": expire_date,
        "stock": stock,
        "description": description,
    }, None


# ── 启动入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    application = create_app()
    application.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
