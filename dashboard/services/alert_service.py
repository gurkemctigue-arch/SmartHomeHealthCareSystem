"""Alert queries and acknowledgement workflow."""

from datetime import datetime

from database.db import get_db


ALERT_LEVELS = {"danger", "warning", "info", "success"}
ALERT_STATUSES = {"open", "acknowledged"}


def get_latest_alerts(limit=5, status="all", level="all"):
    """Return persisted alerts, newest first, with optional filters."""
    limit = max(1, min(int(limit), 100))
    clauses = []
    values = []

    if status in ALERT_STATUSES:
        clauses.append("status = ?")
        values.append(status)
    if level in ALERT_LEVELS:
        clauses.append("level = ?")
        values.append(level)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    values.append(limit)
    rows = get_db().execute(
        "SELECT id, level, title, content, status, acknowledged_at, created_at "
        f"FROM alert_record {where} ORDER BY created_at DESC, id DESC LIMIT ?",
        values,
    ).fetchall()

    return [
        {
            **dict(row),
            "time": _display_time(row["created_at"]),
        }
        for row in rows
    ]


def acknowledge_alert(alert_id):
    """Mark one open alert as acknowledged."""
    db = get_db()
    cursor = db.execute(
        "UPDATE alert_record "
        "SET status = 'acknowledged', acknowledged_at = datetime('now', 'localtime') "
        "WHERE id = ? AND status = 'open'",
        (alert_id,),
    )
    db.commit()
    if cursor.rowcount:
        return True
    return db.execute("SELECT 1 FROM alert_record WHERE id = ?", (alert_id,)).fetchone() is not None


def create_alert(level, title, content):
    """Persist a de-duplicated open alert."""
    normalized_level = level if level in ALERT_LEVELS else "info"
    db = get_db()
    existing = db.execute(
        "SELECT id FROM alert_record "
        "WHERE status = 'open' AND level = ? AND title = ? AND content = ? LIMIT 1",
        (normalized_level, title, content),
    ).fetchone()
    if existing:
        return existing["id"]

    cursor = db.execute(
        "INSERT INTO alert_record (level, title, content, status) VALUES (?, ?, ?, 'open')",
        (normalized_level, title, content),
    )
    db.commit()
    return cursor.lastrowid


def check_alerts(medicine=None, emotion=None):
    """Build deterministic alert candidates from medicine and emotion data."""
    alerts = []
    today = datetime.now().strftime("%Y-%m-%d")

    if medicine:
        expire_date = medicine.get("expire_date")
        if expire_date and expire_date < today:
            alerts.append({
                "level": "danger",
                "title": "药品过期",
                "content": f"{medicine.get('name', '未知药品')}已过期，请勿服用",
            })
        if int(medicine.get("stock") or 0) <= 2:
            alerts.append({
                "level": "warning",
                "title": "库存不足",
                "content": f"{medicine.get('name', '未知药品')}库存不足",
            })

    if emotion and emotion.get("emotion") in {"悲伤", "焦虑", "愤怒"}:
        alerts.append({
            "level": "warning",
            "title": "情绪趋势提醒",
            "content": "检测到连续异常情绪，请家属关注",
        })

    return alerts


def sync_medicine_alerts(medicine):
    """Synchronize open inventory alerts after a medicine is saved."""
    alerts = check_alerts(medicine=medicine)
    active_titles = {alert["title"] for alert in alerts}
    db = get_db()
    inventory_titles = {"药品过期", "库存不足"}
    resolved_titles = inventory_titles - active_titles
    if resolved_titles:
        placeholders = ", ".join("?" for _ in resolved_titles)
        db.execute(
            "UPDATE alert_record "
            "SET status = 'acknowledged', acknowledged_at = datetime('now', 'localtime') "
            f"WHERE status = 'open' AND title IN ({placeholders}) AND content LIKE ?",
            [*sorted(resolved_titles), f"{medicine['name']}%"],
        )
        db.commit()

    for alert in alerts:
        create_alert(alert["level"], alert["title"], alert["content"])


def resolve_medicine_alerts(medicine_name):
    """Close open inventory alerts when a medicine is deleted."""
    db = get_db()
    db.execute(
        "UPDATE alert_record "
        "SET status = 'acknowledged', acknowledged_at = datetime('now', 'localtime') "
        "WHERE status = 'open' AND title IN ('药品过期', '库存不足') AND content LIKE ?",
        (f"{medicine_name}%",),
    )
    db.commit()


def _display_time(value):
    if not value:
        return "--"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)
