"""
多模态智能医疗家庭助手·成果大屏
Flask 主程序
"""
import logging

from flask import Flask, render_template, Response, jsonify, request

from config import Config, setup_logging
from database.db import init_db
from services.video_service import generate_video_stream, get_detect_mode, set_detect_mode
from services.stats_service import get_overview_data, get_medicine_stats, get_trend_data
from services.alert_service import get_latest_alerts
from models.llm_client import ask_local_llm

# 初始化日志
setup_logging()
logger = logging.getLogger(__name__)


def create_app():
    """创建 Flask 应用实例"""
    app = Flask(__name__)
    app.config.from_object(Config)

    # 初始化数据库
    init_db(app)

    # ── 页面路由 ──────────────────────────────────────────

    @app.route("/")
    def index():
        """返回大屏首页"""
        return render_template("index.html")

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
        return jsonify({
            "code": 200,
            "message": "success",
            "data": get_latest_alerts(limit=5)
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


# ── 启动入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    application = create_app()
    application.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
