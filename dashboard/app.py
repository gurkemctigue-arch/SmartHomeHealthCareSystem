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
from wenda import get_engine

setup_logging()
logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    init_db(app)

    # ── 页面 ──

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/chat")
    def chat_page():
        return render_template("chat.html")

    # ── 视频流 ──

    @app.route("/video_feed")
    def video_feed():
        return Response(
            generate_video_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame")

    # ── 数据 API ──

    @app.route("/api/overview")
    def overview():
        return jsonify({"code": 200, "message": "success",
                        "data": get_overview_data()})

    @app.route("/api/stats/medicine")
    def medicine_stats():
        return jsonify({"code": 200, "message": "success",
                        "data": get_medicine_stats()})

    @app.route("/api/stats/trend")
    def trend_stats():
        return jsonify({"code": 200, "message": "success",
                        "data": get_trend_data()})

    @app.route("/api/alerts/latest")
    def latest_alerts():
        return jsonify({"code": 200, "message": "success",
                        "data": get_latest_alerts(limit=5)})

    # ── 检测模式 ──

    @app.route("/api/mode", methods=["GET", "POST"])
    def detect_mode():
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            mode = body.get("mode", "both")
            if mode in ("medicine", "emotion", "both"):
                set_detect_mode(mode)
                return jsonify({"code": 200, "message": "ok", "data": {"mode": mode}})
            return jsonify({"code": 400, "message": "invalid mode"}), 400
        return jsonify({"code": 200, "message": "ok",
                        "data": {"mode": get_detect_mode()}})

    # ── 问答 API (wenda) ──

    @app.route("/api/chat", methods=["POST"])
    def chat():
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()
        if not question:
            return jsonify({"code": 400, "message": "问题不能为空"}), 400

        raw_sid = body.get("session_id")
        sid = raw_sid.strip() if raw_sid else None
        engine = get_engine()
        result = engine.ask(question, session_id=sid)

        try:
            from database.db import get_db
            db = get_db()
            db.execute("INSERT INTO chat_record (question, answer) VALUES (?, ?)",
                       (question, result["answer"]))
            db.commit()
        except Exception as e:
            logger.warning("聊天记录保存失败: %s", e)

        return jsonify({"code": 200, "message": "success", "data": result})

    @app.route("/api/kb/info")
    def kb_info():
        return jsonify({"code": 200, "message": "success",
                        "data": get_engine().info()})

    @app.route("/api/chat/history")
    def chat_history():
        try:
            from database.db import get_db
            db = get_db()
            rows = db.execute(
                "SELECT question, answer, created_at FROM chat_record "
                "ORDER BY created_at DESC LIMIT 50").fetchall()
            return jsonify({"code": 200, "message": "success", "data": [
                {"question": r["question"], "answer": r["answer"],
                 "time": r["created_at"]} for r in reversed(rows)]})
        except Exception as e:
            logger.warning("历史获取失败: %s", e)
            return jsonify({"code": 200, "message": "success", "data": []})

    @app.route("/api/chat/suggestions")
    def chat_suggestions():
        return jsonify({"code": 200, "message": "success", "data": [
            "布洛芬的用法用量和注意事项是什么？",
            "阿莫西林和阿司匹林可以一起吃吗？",
            "高血压患者日常饮食需要注意什么？",
            "感冒药和退烧药需要间隔多久服用？",
            "家庭药箱应该常备哪些药品？"]})

    @app.route("/api/session/<sid>", methods=["GET", "DELETE"])
    def session_api(sid):
        engine = get_engine()
        if request.method == "DELETE":
            ok = engine.clear(sid)
            return jsonify({"code": 200 if ok else 404,
                            "message": "已清除" if ok else "不存在"})
        h = engine.history(sid)
        return jsonify({"code": 200, "message": "success",
                        "data": {"session_id": sid, "history": h,
                                 "turn_count": len(h) // 2}})

    # ── 语音播报 TTS ──

    @app.route("/api/tts", methods=["POST"])
    def tts():
        """文本转语音，返回 MP3 音频流"""
        import asyncio

        body = request.get_json(silent=True) or {}
        text = (body.get("text") or "").strip()
        if not text:
            return jsonify({"code": 400, "message": "文本不能为空"}), 400

        # 截断过长文本（edge-tts 单次有字数限制）
        if len(text) > 500:
            text = text[:500]

        voice = "zh-CN-YunxiNeural"  # 标准男声

        try:
            # edge-tts 是异步库，用 asyncio 在同步端点中调用
            result = asyncio.run(_tts_generate(text, voice))
            return Response(result, mimetype="audio/mpeg")
        except Exception as e:
            logger.warning("TTS 生成失败: %s", e)
            return jsonify({"code": 500, "message": f"语音生成失败: {e}"}), 500


    async def _tts_generate(text: str, voice: str) -> bytes:
        import edge_tts
        import io
        communicate = edge_tts.Communicate(text, voice)
        mp3_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data.write(chunk["data"])
        return mp3_data.getvalue()

    # ── 健康检查 ──

    @app.route("/api/health")
    def health():
        import platform
        status = {
            "service": "多模态智能医疗家庭助手",
            "status": "running",
            "python": platform.python_version(),
        }
        try:
            info = get_engine().info()
            status["llm_available"] = info["ready"]
            status["llm_backend"] = info["backend"]
            status["rag_status"] = info["rag"]
        except Exception:
            status["llm_available"] = False
            status["llm_backend"] = "error"
        try:
            from services.video_service import detector as d
            status["detector"] = "YOLO-OBB" if d.is_real_model else "MOCK"
        except Exception:
            status["detector"] = "unavailable"
        try:
            from services.video_service import emotion_model as e
            status["emotion"] = "CNN" if e.is_real_model else "MOCK"
        except Exception:
            status["emotion"] = "unavailable"
        try:
            from database.db import get_db
            get_db().execute("SELECT 1")
            status["database"] = "connected"
        except Exception:
            status["database"] = "disconnected"
        return jsonify({"code": 200, "message": "ok", "data": status})

    # ── 错误处理 ──

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"code": 404, "message": "接口不存在"}), 404

    @app.errorhandler(500)
    def server_error(_e):
        return jsonify({"code": 500, "message": "服务器内部错误"}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    # 启动时后台预热引擎
    import threading
    def _warmup():
        logger.info("后台预热问答引擎...")
        get_engine()
    threading.Thread(target=_warmup, daemon=True).start()
    application.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)
