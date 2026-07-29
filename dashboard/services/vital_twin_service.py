"""Family health data and deterministic signals for the Vital Twin workspace."""

import json
from datetime import date, datetime, timedelta

from database.db import get_db


class VitalTwinValidationError(ValueError):
    pass


class VitalTwinNotFoundError(LookupError):
    pass


RELATIONS = {
    "self": "本人",
    "spouse": "伴侣",
    "father": "父亲",
    "mother": "母亲",
    "child": "子女",
    "grandparent": "长辈",
    "other": "家人",
}
SEXES = {"unspecified", "female", "male"}
MEMBER_COLORS = {"mint", "cyan", "coral", "amber", "violet"}
METRICS = {
    "blood_pressure": {"label": "血压", "unit": "mmHg", "icon": "heart-pulse"},
    "heart_rate": {"label": "心率", "unit": "bpm", "icon": "activity"},
    "spo2": {"label": "血氧", "unit": "%", "icon": "wind"},
    "temperature": {"label": "体温", "unit": "°C", "icon": "thermometer"},
    "blood_glucose": {"label": "血糖", "unit": "mmol/L", "icon": "droplets"},
    "weight": {"label": "体重", "unit": "kg", "icon": "scale"},
}
METRIC_RANGES = {
    "blood_pressure": ((40, 300), (30, 200)),
    "heart_rate": ((25, 250), None),
    "spo2": ((50, 100), None),
    "temperature": ((30, 45), None),
    "blood_glucose": ((1, 40), None),
    "weight": ((1, 400), None),
}


def seed_vital_twin(conn):
    """Create a truthful empty primary profile for first-time use."""
    count = conn.execute("SELECT COUNT(*) FROM family_member").fetchone()[0]
    if count:
        return
    conn.execute(
        "INSERT INTO family_member "
        "(name, relation, sex, color, conditions, allergies, is_primary) "
        "VALUES (?, 'self', 'unspecified', 'mint', '[]', '[]', 1)",
        ("我",),
    )


def list_members():
    rows = get_db().execute(
        "SELECT * FROM family_member ORDER BY is_primary DESC, id ASC"
    ).fetchall()
    return [_serialize_member(row, include_status=True) for row in rows]


def create_member(payload):
    data = _validate_member(payload)
    db = get_db()
    cursor = db.execute(
        "INSERT INTO family_member "
        "(name, relation, sex, birth_date, height_cm, color, conditions, allergies, is_primary) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (
            data["name"], data["relation"], data["sex"], data["birth_date"],
            data["height_cm"], data["color"], json.dumps(data["conditions"], ensure_ascii=False),
            json.dumps(data["allergies"], ensure_ascii=False),
        ),
    )
    db.commit()
    return get_member(cursor.lastrowid)


def update_member(member_id, payload):
    _require_member(member_id)
    data = _validate_member(payload)
    db = get_db()
    db.execute(
        "UPDATE family_member SET name = ?, relation = ?, sex = ?, birth_date = ?, "
        "height_cm = ?, color = ?, conditions = ?, allergies = ?, "
        "updated_at = datetime('now', 'localtime') WHERE id = ?",
        (
            data["name"], data["relation"], data["sex"], data["birth_date"],
            data["height_cm"], data["color"], json.dumps(data["conditions"], ensure_ascii=False),
            json.dumps(data["allergies"], ensure_ascii=False), member_id,
        ),
    )
    db.commit()
    return get_member(member_id)


def delete_member(member_id):
    member = _require_member(member_id)
    db = get_db()
    total = db.execute("SELECT COUNT(*) FROM family_member").fetchone()[0]
    if total <= 1:
        raise VitalTwinValidationError("家庭中至少需要保留一位成员")
    if member["is_primary"]:
        replacement = db.execute(
            "SELECT id FROM family_member WHERE id <> ? ORDER BY id LIMIT 1", (member_id,)
        ).fetchone()
        db.execute("UPDATE family_member SET is_primary = 1 WHERE id = ?", (replacement["id"],))
    db.execute("DELETE FROM family_member WHERE id = ?", (member_id,))
    db.commit()
    return {"id": member_id}


def get_member(member_id):
    row = _require_member(member_id)
    return _serialize_member(row, include_status=True)


def add_measurement(payload):
    if not isinstance(payload, dict):
        raise VitalTwinValidationError("测量内容必须是 JSON 对象")
    try:
        member_id = int(payload.get("member_id"))
    except (TypeError, ValueError):
        raise VitalTwinValidationError("请选择家庭成员") from None
    _require_member(member_id)
    metric_type = str(payload.get("metric_type") or "").strip()
    if metric_type not in METRICS:
        raise VitalTwinValidationError("不支持该测量类型")
    try:
        primary = float(payload.get("value_primary"))
    except (TypeError, ValueError):
        raise VitalTwinValidationError("请输入有效测量值") from None
    primary_range, secondary_range = METRIC_RANGES[metric_type]
    if not primary_range[0] <= primary <= primary_range[1]:
        raise VitalTwinValidationError(f"{METRICS[metric_type]['label']}测量值超出可记录范围")
    secondary = None
    if metric_type == "blood_pressure":
        try:
            secondary = float(payload.get("value_secondary"))
        except (TypeError, ValueError):
            raise VitalTwinValidationError("请输入舒张压") from None
        if not secondary_range[0] <= secondary <= secondary_range[1]:
            raise VitalTwinValidationError("舒张压超出可记录范围")
        if secondary >= primary:
            raise VitalTwinValidationError("收缩压应高于舒张压")
    measured_at = _parse_datetime(payload.get("measured_at"))
    note = str(payload.get("note") or "").strip()
    if len(note) > 160:
        raise VitalTwinValidationError("备注不能超过 160 个字符")
    db = get_db()
    cursor = db.execute(
        "INSERT INTO health_measurement "
        "(member_id, metric_type, value_primary, value_secondary, unit, source, note, measured_at) "
        "VALUES (?, ?, ?, ?, ?, 'manual', ?, ?)",
        (member_id, metric_type, primary, secondary, METRICS[metric_type]["unit"], note, measured_at),
    )
    db.commit()
    row = db.execute("SELECT * FROM health_measurement WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return _serialize_measurement(row)


def create_task(payload):
    if not isinstance(payload, dict):
        raise VitalTwinValidationError("任务内容必须是 JSON 对象")
    try:
        member_id = int(payload.get("member_id"))
    except (TypeError, ValueError):
        raise VitalTwinValidationError("请选择家庭成员") from None
    _require_member(member_id)
    title = str(payload.get("title") or "").strip()
    if not title or len(title) > 60:
        raise VitalTwinValidationError("任务名称需要 1 到 60 个字符")
    detail = str(payload.get("detail") or "").strip()
    if len(detail) > 180:
        raise VitalTwinValidationError("任务说明不能超过 180 个字符")
    priority = str(payload.get("priority") or "normal")
    if priority not in {"normal", "important", "urgent"}:
        raise VitalTwinValidationError("任务优先级无效")
    due_at = _parse_datetime(payload.get("due_at"), allow_future=True)
    db = get_db()
    cursor = db.execute(
        "INSERT INTO health_task (member_id, task_type, title, detail, due_at, priority) "
        "VALUES (?, 'care', ?, ?, ?, ?)",
        (member_id, title, detail, due_at, priority),
    )
    db.commit()
    return {"id": cursor.lastrowid, "member_id": member_id, "title": title}


def complete_task(task_id):
    db = get_db()
    cursor = db.execute(
        "UPDATE health_task SET status = 'completed', "
        "completed_at = datetime('now', 'localtime') WHERE id = ? AND status = 'open'",
        (task_id,),
    )
    if cursor.rowcount == 0:
        row = db.execute("SELECT id FROM health_task WHERE id = ?", (task_id,)).fetchone()
        if not row:
            raise VitalTwinNotFoundError("照护任务不存在")
    db.commit()
    return {"id": task_id, "status": "completed"}


def get_vital_twin_overview(member_id=None):
    members = list_members()
    if not members:
        raise VitalTwinNotFoundError("尚未建立家庭档案")
    try:
        selected_id = int(member_id) if member_id is not None else members[0]["id"]
    except (TypeError, ValueError):
        raise VitalTwinValidationError("家庭成员编号无效") from None
    member = get_member(selected_id)
    latest = _latest_measurements(selected_id)
    signals = _body_signals(latest)
    actions = _build_actions(selected_id, latest)
    timeline = _build_timeline(selected_id)
    assessed = list(latest.values())
    if assessed:
        penalty = sum(16 if item["status"] == "danger" else 7 if item["status"] == "warning" else 0 for item in assessed)
        score = max(42, 96 - penalty)
    else:
        score = None
    return {
        "member": member,
        "members": members,
        "score": score,
        "coverage": round(len(latest) / len(METRICS) * 100),
        "signals": signals,
        "latest_measurements": list(latest.values()),
        "actions": actions[:6],
        "timeline": timeline,
        "metrics": [
            {"key": key, **meta} for key, meta in METRICS.items()
        ],
        "updated_at": datetime.now().isoformat(timespec="minutes"),
        "disclaimer": "生命镜像用于家庭健康记录与趋势观察，不替代医学诊断。",
    }


def _validate_member(payload):
    if not isinstance(payload, dict):
        raise VitalTwinValidationError("成员内容必须是 JSON 对象")
    name = str(payload.get("name") or "").strip()
    if not name or len(name) > 24:
        raise VitalTwinValidationError("成员称呼需要 1 到 24 个字符")
    relation = str(payload.get("relation") or "other")
    if relation not in RELATIONS:
        raise VitalTwinValidationError("成员关系无效")
    sex = str(payload.get("sex") or "unspecified")
    if sex not in SEXES:
        raise VitalTwinValidationError("性别选项无效")
    color = str(payload.get("color") or "mint")
    if color not in MEMBER_COLORS:
        raise VitalTwinValidationError("档案颜色无效")
    birth_date = payload.get("birth_date") or None
    if birth_date:
        try:
            parsed_date = datetime.strptime(str(birth_date), "%Y-%m-%d").date()
        except ValueError:
            raise VitalTwinValidationError("出生日期格式必须为 YYYY-MM-DD") from None
        if parsed_date > date.today() or parsed_date.year < 1900:
            raise VitalTwinValidationError("出生日期不在有效范围内")
        birth_date = parsed_date.isoformat()
    height_cm = payload.get("height_cm") or None
    if height_cm is not None:
        try:
            height_cm = round(float(height_cm), 1)
        except (TypeError, ValueError):
            raise VitalTwinValidationError("身高必须是数字") from None
        if not 40 <= height_cm <= 250:
            raise VitalTwinValidationError("身高需要在 40 到 250 厘米之间")
    conditions = _string_list(payload.get("conditions"), "健康状况")
    allergies = _string_list(payload.get("allergies"), "过敏信息")
    return {
        "name": name, "relation": relation, "sex": sex, "birth_date": birth_date,
        "height_cm": height_cm, "color": color, "conditions": conditions,
        "allergies": allergies,
    }


def _string_list(value, field):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 12:
        raise VitalTwinValidationError(f"{field}格式无效")
    result = []
    for item in value:
        text = str(item).strip()
        if text and len(text) <= 40 and text not in result:
            result.append(text)
    return result


def _parse_datetime(value, allow_future=False):
    if not value:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip().replace("T", " ")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise VitalTwinValidationError("时间格式无效") from None
    if not allow_future and parsed > datetime.now() + timedelta(minutes=5):
        raise VitalTwinValidationError("记录时间不能晚于当前时间")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _require_member(member_id):
    row = get_db().execute("SELECT * FROM family_member WHERE id = ?", (member_id,)).fetchone()
    if not row:
        raise VitalTwinNotFoundError("家庭成员不存在")
    return row


def _serialize_member(row, include_status=False):
    result = dict(row)
    result["conditions"] = _json_list(result.get("conditions"))
    result["allergies"] = _json_list(result.get("allergies"))
    result["relation_label"] = RELATIONS.get(result.get("relation"), "家人")
    result["initial"] = (result.get("name") or "家")[0].upper()
    result["age"] = _age(result.get("birth_date"))
    result["is_primary"] = bool(result.get("is_primary"))
    if include_status:
        row_status = get_db().execute(
            "SELECT measured_at FROM health_measurement WHERE member_id = ? "
            "ORDER BY measured_at DESC, id DESC LIMIT 1", (result["id"],)
        ).fetchone()
        result["last_measurement_at"] = row_status["measured_at"] if row_status else None
        result["status"] = "ready" if row_status else "baseline"
    return result


def _json_list(value):
    try:
        parsed = json.loads(value or "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def _age(birth_date):
    if not birth_date:
        return None
    born = datetime.strptime(birth_date, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _latest_measurements(member_id):
    rows = get_db().execute(
        "SELECT hm.* FROM health_measurement hm WHERE hm.member_id = ? "
        "AND hm.id = (SELECT recent.id FROM health_measurement recent "
        "WHERE recent.member_id = hm.member_id AND recent.metric_type = hm.metric_type "
        "ORDER BY recent.measured_at DESC, recent.id DESC LIMIT 1) "
        "ORDER BY hm.measured_at DESC, hm.id DESC", (member_id,)
    ).fetchall()
    return {row["metric_type"]: _serialize_measurement(row) for row in rows}


def _serialize_measurement(row):
    item = dict(row)
    item.update(METRICS[item["metric_type"]])
    item["status"], item["status_label"] = _classify_measurement(
        item["metric_type"], item["value_primary"], item.get("value_secondary")
    )
    primary = _format_number(item["value_primary"])
    secondary = _format_number(item.get("value_secondary"))
    item["display_value"] = f"{primary}/{secondary}" if secondary is not None else str(primary)
    return item


def _format_number(value):
    if value is None:
        return None
    return int(value) if float(value).is_integer() else round(float(value), 1)


def _classify_measurement(metric_type, primary, secondary=None):
    if metric_type == "blood_pressure":
        if primary >= 180 or secondary >= 120:
            return "danger", "需要立即关注"
        if primary >= 140 or secondary >= 90 or primary < 90 or secondary < 60:
            return "warning", "偏离常见范围"
    elif metric_type == "heart_rate":
        if primary > 120 or primary < 45:
            return "danger", "需要立即关注"
        if primary > 100 or primary < 55:
            return "warning", "偏离常见范围"
    elif metric_type == "spo2":
        if primary < 90:
            return "danger", "需要立即关注"
        if primary < 95:
            return "warning", "低于常见范围"
    elif metric_type == "temperature":
        if primary >= 39 or primary < 35:
            return "danger", "需要立即关注"
        if primary >= 37.5:
            return "warning", "高于常见范围"
    elif metric_type == "blood_glucose":
        if primary > 13.9 or primary < 3:
            return "danger", "需要立即关注"
        if primary > 7.8 or primary < 3.9:
            return "warning", "偏离常见范围"
    return "good", "记录正常"


def _body_signals(latest):
    groups = {
        "mind": ("神经与情绪", "head", []),
        "cardio": ("心血管", "chest", ["blood_pressure", "heart_rate"]),
        "respiratory": ("呼吸与血氧", "lungs", ["spo2"]),
        "metabolic": ("代谢状态", "abdomen", ["blood_glucose", "weight"]),
        "thermal": ("体温状态", "body", ["temperature"]),
    }
    result = []
    rank = {"danger": 3, "warning": 2, "good": 1, "baseline": 0}
    for key, (label, region, metric_keys) in groups.items():
        values = [latest[item] for item in metric_keys if item in latest]
        status = max((item["status"] for item in values), key=lambda value: rank[value], default="baseline")
        result.append({
            "key": key, "label": label, "region": region, "status": status,
            "summary": " · ".join(f"{item['label']} {item['display_value']}{item['unit']}" for item in values)
            if values else "等待建立个人基线",
            "measurement_keys": metric_keys,
        })
    return result


def _build_actions(member_id, latest):
    actions = []
    severity = {"danger": 0, "urgent": 0, "warning": 1, "important": 1, "info": 2, "normal": 2}
    for item in latest.values():
        if item["status"] in {"danger", "warning"}:
            actions.append({
                "id": f"measurement-{item['id']}", "kind": "measurement",
                "priority": item["status"], "icon": item["icon"],
                "title": f"{item['label']}{item['status_label']}",
                "detail": f"最近记录 {item['display_value']} {item['unit']}，建议复测并结合个人情况判断。",
                "action": "record", "metric_type": item["metric_type"],
            })
    db = get_db()
    task_rows = db.execute(
        "SELECT * FROM health_task WHERE member_id = ? AND status = 'open' "
        "ORDER BY CASE priority WHEN 'urgent' THEN 0 WHEN 'important' THEN 1 ELSE 2 END, due_at ASC LIMIT 8",
        (member_id,),
    ).fetchall()
    for row in task_rows:
        actions.append({
            "id": f"task-{row['id']}", "task_id": row["id"], "kind": "task",
            "priority": row["priority"], "icon": "clipboard-check", "title": row["title"],
            "detail": row["detail"] or f"计划时间 {row['due_at']}", "action": "complete",
        })
    required = ["blood_pressure", "spo2", "weight"]
    missing = [key for key in required if key not in latest]
    if missing:
        first = missing[0]
        actions.append({
            "id": "baseline", "kind": "baseline", "priority": "info", "icon": "scan-heart",
            "title": "建立个人健康基线", "detail": f"先记录{METRICS[first]['label']}，生命镜像才能形成可靠趋势。",
            "action": "record", "metric_type": first,
        })
    medicine = db.execute(
        "SELECT id, name, stock, expire_date FROM medicine "
        "WHERE stock <= 5 OR (expire_date IS NOT NULL AND date(expire_date) <= date('now', '+30 days')) "
        "ORDER BY stock ASC, expire_date ASC LIMIT 2"
    ).fetchall()
    for row in medicine:
        expired = row["expire_date"] and row["expire_date"] < date.today().isoformat()
        actions.append({
            "id": f"medicine-{row['id']}", "kind": "medicine",
            "priority": "danger" if expired else "warning", "icon": "pill",
            "title": f"家庭药箱 · {row['name']}",
            "detail": "药品已过期，请停止使用并妥善处理。" if expired else f"当前库存 {row['stock']}，建议检查补充或有效期。",
            "action": "medicine",
        })
    open_alerts = db.execute("SELECT COUNT(*) FROM alert_record WHERE status = 'open'").fetchone()[0]
    if open_alerts:
        actions.append({
            "id": "family-alerts", "kind": "alert", "priority": "warning", "icon": "bell-ring",
            "title": f"{open_alerts} 条家庭告警待处理", "detail": "核对药品、检测和系统告警的处置状态。",
            "action": "alerts",
        })
    actions.sort(key=lambda item: severity.get(item["priority"], 3))
    return actions


def _build_timeline(member_id):
    db = get_db()
    rows = db.execute(
        "SELECT id, metric_type AS event_type, value_primary, value_secondary, unit, "
        "measured_at AS event_at FROM health_measurement WHERE member_id = ? "
        "ORDER BY measured_at DESC, id DESC LIMIT 18", (member_id,)
    ).fetchall()
    events = []
    for row in rows:
        serialized = _serialize_measurement({
            "id": row["id"], "member_id": member_id, "metric_type": row["event_type"],
            "value_primary": row["value_primary"], "value_secondary": row["value_secondary"],
            "unit": row["unit"], "source": "manual", "note": "", "measured_at": row["event_at"],
        })
        events.append({
            "id": f"measurement-{row['id']}", "type": "measurement", "metric_type": row["event_type"],
            "title": f"记录{serialized['label']}",
            "detail": f"{serialized['display_value']} {serialized['unit']}",
            "status": serialized["status"], "event_at": row["event_at"], "icon": serialized["icon"],
        })
    task_rows = db.execute(
        "SELECT id, title, detail, priority, status, COALESCE(completed_at, due_at) AS event_at "
        "FROM health_task WHERE member_id = ? ORDER BY event_at DESC LIMIT 12", (member_id,)
    ).fetchall()
    for row in task_rows:
        events.append({
            "id": f"task-{row['id']}", "type": "task", "title": row["title"],
            "detail": row["detail"] or ("已完成" if row["status"] == "completed" else "待处理"),
            "status": "good" if row["status"] == "completed" else "warning",
            "event_at": row["event_at"], "icon": "clipboard-check",
        })
    events.sort(key=lambda item: item["event_at"] or "", reverse=True)
    return events[:24]
