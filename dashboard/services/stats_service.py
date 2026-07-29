"""数据统计服务 — 从 SQLite 数据库实时查询"""

import logging
from datetime import datetime, timedelta

from database.db import get_db

logger = logging.getLogger(__name__)


def get_overview_data():
    """获取顶部概览数据（从数据库实时查询）"""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    db = get_db()

    # 今日检测次数
    today_count = db.execute(
        "SELECT COUNT(*) FROM detection_record "
        "WHERE date(created_at) = ? AND trim(COALESCE(medicine_name, '')) <> ''",
        (today_str,)
    ).fetchone()[0]

    # 昨日检测次数（用于计算趋势）
    yesterday_count = db.execute(
        "SELECT COUNT(*) FROM detection_record "
        "WHERE date(created_at) = ? AND trim(COALESCE(medicine_name, '')) <> ''",
        (yesterday_str,)
    ).fetchone()[0]

    # 今日告警数
    alert_count = db.execute(
        "SELECT COUNT(*) FROM alert_record WHERE date(created_at) = ? AND status = 'open'",
        (today_str,)
    ).fetchone()[0]

    # 昨日告警数
    yesterday_alert = db.execute(
        "SELECT COUNT(*) FROM alert_record WHERE date(created_at) = ? AND status = 'open'",
        (yesterday_str,)
    ).fetchone()[0]

    # 变化率（避免除零）
    def _rate(today, yesterday):
        if yesterday == 0:
            return 0 if today == 0 else 100
        return round((today - yesterday) / yesterday * 100)

    # LLM 状态（从模型客户端获取真实状态）
    try:
        from models.llm_client import get_llm_backend, get_rag_status, is_llm_available
        llm_backend = get_llm_backend()
        rag_status = get_rag_status()
        llm_ok = is_llm_available()
    except Exception:
        llm_backend = "未检测"
        rag_status = "未检测"
        llm_ok = False

    return {
        "today_count": today_count,
        "today_count_rate": _rate(today_count, yesterday_count),
        "alert_count": alert_count,
        "alert_rate": _rate(alert_count, yesterday_alert),
        "llm_status": "已连接" if llm_ok else "未连接",
        "llm_desc": f"{llm_backend} · {rag_status}",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
    }


def get_medicine_stats():
    """获取药品类别统计（从检测记录表聚合）"""
    db = get_db()
    today_str = datetime.now().strftime("%Y-%m-%d")

    try:
        rows = db.execute("""
            SELECT
                COALESCE(
                    (SELECT category FROM medicine WHERE medicine.name = detection_record.medicine_name),
                    '其他'
                ) AS category,
                COUNT(*) AS cnt
            FROM detection_record
            WHERE date(created_at) = ?
              AND trim(COALESCE(medicine_name, '')) <> ''
            GROUP BY category
            ORDER BY cnt DESC
            LIMIT 8
        """, (today_str,)).fetchall()

        if rows:
            return [{"name": r["category"], "value": r["cnt"]} for r in rows]
    except Exception as e:
        logger.warning("药品类别统计查询失败: %s", e)

    # 数据库为空时返回空列表（前端显示"暂无数据"）
    return []


def get_trend_data():
    """获取近7日检测趋势（从数据库实时查询）"""
    db = get_db()
    today = datetime.now()
    dates = []
    values = []

    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        dates.append(d.strftime("%m-%d"))

        count = db.execute(
            "SELECT COUNT(*) FROM detection_record "
            "WHERE date(created_at) = ? AND trim(COALESCE(medicine_name, '')) <> ''",
            (date_str,)
        ).fetchone()[0]
        values.append(count)

    return {
        "dates": dates,
        "values": values
    }


def get_latest_detections(limit=20):
    """Return recent valid medicine detections for the detection workspace."""
    limit = max(1, min(int(limit), 100))
    rows = get_db().execute(
        "SELECT id, medicine_name, confidence, emotion, emotion_confidence, created_at "
        "FROM detection_record "
        "WHERE trim(COALESCE(medicine_name, '')) <> '' "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_emotion_summary(days=7):
    """Return the latest usable emotion and its recent distribution."""
    days = max(1, min(int(days), 30))
    db = get_db()
    valid_emotion = (
        "trim(COALESCE(emotion, '')) <> '' "
        "AND emotion NOT IN ('检测中...', '已关闭') "
        "AND COALESCE(emotion_confidence, 0) > 0"
    )
    latest = db.execute(
        "SELECT emotion, emotion_confidence, created_at FROM detection_record "
        f"WHERE {valid_emotion} ORDER BY created_at DESC, id DESC LIMIT 1"
    ).fetchone()
    distribution = db.execute(
        "SELECT emotion AS name, COUNT(*) AS value FROM detection_record "
        f"WHERE {valid_emotion} AND created_at >= datetime('now', ?, 'localtime') "
        "GROUP BY emotion ORDER BY value DESC",
        (f"-{days} days",),
    ).fetchall()
    return {
        "latest": dict(latest) if latest else None,
        "distribution": [dict(row) for row in distribution],
    }
