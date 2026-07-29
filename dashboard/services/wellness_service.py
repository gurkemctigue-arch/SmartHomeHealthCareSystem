"""Health-profile storage and deterministic climate wellness recommendations."""

from datetime import date, datetime, timezone
import hashlib
import json

from database.db import get_db
from services.recipe_service import get_recipe_summary


AGE_GROUPS = {"child", "teen", "adult", "older_adult"}
SEXES = {"female", "male", "unspecified"}
CONDITIONS = {
    "hypertension", "diabetes", "hyperlipidemia", "gout", "kidney_disease",
    "cardiovascular", "asthma", "allergic_rhinitis",
}
ALLERGIES = {"dairy", "egg", "nuts", "seafood", "soy", "gluten"}
DIETS = {"vegetarian", "vegan", "low_sodium", "low_sugar"}
GOALS = {"weight_control", "sleep", "immunity", "blood_pressure", "blood_sugar"}

DEFAULT_PROFILE = {
    "age_group": "adult",
    "sex": "unspecified",
    "conditions": [],
    "allergies": [],
    "medications": "",
    "dietary_preferences": [],
    "health_goals": [],
    "retain_location": False,
    "updated_at": None,
}


class ProfileValidationError(ValueError):
    pass


def get_health_profile():
    row = get_db().execute("SELECT * FROM health_profile WHERE id = 1").fetchone()
    if not row:
        return dict(DEFAULT_PROFILE)
    return {
        "age_group": row["age_group"],
        "sex": row["sex"],
        "conditions": _json_list(row["conditions"]),
        "allergies": _json_list(row["allergies"]),
        "medications": row["medications"] or "",
        "dietary_preferences": _json_list(row["dietary_preferences"]),
        "health_goals": _json_list(row["health_goals"]),
        "retain_location": bool(row["retain_location"]),
        "updated_at": row["updated_at"],
    }


def save_health_profile(payload):
    profile = validate_profile(payload)
    db = get_db()
    db.execute(
        "INSERT INTO health_profile (id, age_group, sex, conditions, allergies, medications, "
        "dietary_preferences, health_goals, retain_location, updated_at) "
        "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime')) "
        "ON CONFLICT(id) DO UPDATE SET age_group=excluded.age_group, sex=excluded.sex, "
        "conditions=excluded.conditions, allergies=excluded.allergies, "
        "medications=excluded.medications, dietary_preferences=excluded.dietary_preferences, "
        "health_goals=excluded.health_goals, retain_location=excluded.retain_location, "
        "updated_at=excluded.updated_at",
        (
            profile["age_group"], profile["sex"],
            json.dumps(profile["conditions"], ensure_ascii=False),
            json.dumps(profile["allergies"], ensure_ascii=False),
            profile["medications"],
            json.dumps(profile["dietary_preferences"], ensure_ascii=False),
            json.dumps(profile["health_goals"], ensure_ascii=False),
            int(profile["retain_location"]),
        ),
    )
    db.commit()
    return get_health_profile()


def validate_profile(payload):
    if not isinstance(payload, dict):
        raise ProfileValidationError("健康档案必须是 JSON 对象")
    profile = {
        "age_group": str(payload.get("age_group") or "adult"),
        "sex": str(payload.get("sex") or "unspecified"),
        "conditions": _validated_choices(payload.get("conditions"), CONDITIONS, "慢性病"),
        "allergies": _validated_choices(payload.get("allergies"), ALLERGIES, "过敏项"),
        "dietary_preferences": _validated_choices(payload.get("dietary_preferences"), DIETS, "饮食偏好"),
        "health_goals": _validated_choices(payload.get("health_goals"), GOALS, "健康目标"),
        "medications": str(payload.get("medications") or "").strip(),
        "retain_location": bool(payload.get("retain_location", False)),
    }
    if profile["age_group"] not in AGE_GROUPS:
        raise ProfileValidationError("年龄段选项无效")
    if profile["sex"] not in SEXES:
        raise ProfileValidationError("性别选项无效")
    if len(profile["medications"]) > 300:
        raise ProfileValidationError("用药信息不能超过 300 个字符")
    return profile


def get_daily_recommendation(snapshot, profile=None):
    profile = profile or get_health_profile()
    cache_key = _recommendation_cache_key(snapshot, profile)
    db = get_db()
    cached = db.execute(
        "SELECT payload FROM wellness_recommendation_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    if cached:
        payload = json.loads(cached["payload"])
        payload["meta"]["cache"] = "cache"
        return payload

    recommendation = build_recommendation(snapshot, profile)
    recommendation["meta"]["cache"] = "generated"
    db.execute(
        "INSERT OR REPLACE INTO wellness_recommendation_cache "
        "(cache_key, recommendation_date, payload, created_at) VALUES (?, ?, ?, ?)",
        (
            cache_key,
            date.today().isoformat(),
            json.dumps(recommendation, ensure_ascii=False),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.execute(
        "DELETE FROM wellness_recommendation_cache WHERE recommendation_date < date('now', '-14 days')"
    )
    db.commit()
    return recommendation


def build_recommendation(snapshot, profile):
    current = snapshot["current"]
    today = snapshot["today"]
    temperature = current.get("feels_like") or current.get("temperature") or 20
    aqi = current.get("aqi")
    uv = today.get("uv_index_max") or 0
    humidity = current.get("humidity") or 50
    rain = today.get("precipitation_probability_max") or 0

    risks = _risk_matrix(temperature, aqi, uv, humidity, rain, profile)
    meals = _meal_plan(temperature, profile)
    hydration = _hydration_plan(temperature, profile)
    timeline = _wellness_timeline(snapshot, aqi, uv, rain)
    protections = _protection_plan(temperature, aqi, uv, humidity, profile)

    elevated = [item["label"] for item in risks if item["level"] in ("elevated", "high")]
    focus = "、".join(elevated[:2]) if elevated else "规律作息与均衡饮食"
    city = snapshot["location"].get("name") or "当前位置"
    condition = current.get("condition") or "当前天气"
    briefing = (
        f"{city}今日{condition}，体感 {round(temperature)} 度。"
        f"今日重点关注{focus}。饮食以{meals[1]['title']}为核心搭配，"
        f"补水建议为{hydration['target']}。以上内容仅用于日常健康管理参考。"
    )

    notices = ["推荐用于日常健康管理，不替代医生诊断、处方或个体化营养治疗。"]
    if profile.get("medications"):
        notices.append("已记录当前用药；食物与药物相互作用需向医生或药师核实。")
    if "kidney_disease" in profile.get("conditions", []):
        notices.append("肾脏疾病的水分、钾、磷和蛋白质摄入需依据化验结果与医嘱调整。")
    if profile.get("age_group") in ("child", "older_adult"):
        notices.append("儿童或老年人的能量、吞咽与基础病需求需由照护者进一步确认。")

    return {
        "summary": {
            "title": f"{condition}日的身体节律方案",
            "focus": focus,
            "briefing": briefing,
            "hydration": hydration,
        },
        "meals": meals,
        "timeline": timeline,
        "risks": risks,
        "protections": protections,
        "excluded": _excluded_labels(profile),
        "notices": notices,
        "references": [
            {"name": "中国居民膳食指南（2022）", "url": "https://dg.cnsoc.org/"},
            {"name": "WHO Global Air Quality Guidelines", "url": "https://www.who.int/publications/i/item/9789240034228"},
            {"name": "WHO UV radiation guidance", "url": "https://www.who.int/news-room/questions-and-answers/item/radiation-ultraviolet-(uv)"},
        ],
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rules_version": "2026.07.3",
            "cache": "generated",
        },
    }


def _risk_matrix(temperature, aqi, uv, humidity, rain, profile):
    heat_score = min(100, max(8, round((temperature - 18) * 5))) if temperature >= 18 else min(100, round((18 - temperature) * 6))
    heat_label = "热应激" if temperature >= 18 else "寒冷刺激"
    heat_level = "high" if temperature >= 35 or temperature <= 0 else "elevated" if temperature >= 30 or temperature <= 8 else "low"
    air_score = min(100, round((aqi or 0) / 2))
    air_level = "high" if (aqi or 0) > 150 else "elevated" if (aqi or 0) > 100 else "guarded" if (aqi or 0) > 50 else "low"
    uv_score = min(100, round(uv * 9))
    uv_level = "high" if uv >= 8 else "elevated" if uv >= 6 else "guarded" if uv >= 3 else "low"
    damp_score = min(100, round(max(humidity - 45, 0) * 1.4 + rain * 0.25))
    damp_level = "high" if damp_score >= 75 else "elevated" if damp_score >= 50 else "guarded" if damp_score >= 30 else "low"

    respiratory_sensitive = bool({"asthma", "allergic_rhinitis"} & set(profile.get("conditions", [])))
    if respiratory_sensitive and air_level in ("guarded", "elevated"):
        air_level = "high" if air_level == "elevated" else "elevated"
        air_score = min(100, air_score + 15)

    return [
        {"key": "thermal", "label": heat_label, "value": f"{round(temperature)}°", "score": heat_score, "level": heat_level, "icon": "thermometer-sun", "guidance": "避开体感最极端的时段，分段活动。"},
        {"key": "air", "label": "呼吸负荷", "value": f"US AQI {round(aqi) if aqi is not None else '--'}", "score": air_score, "level": air_level, "icon": "wind", "guidance": "污染升高时缩短户外活动并关注呼吸症状。"},
        {"key": "uv", "label": "紫外线", "value": f"UV {uv:.1f}", "score": uv_score, "level": uv_level, "icon": "sun", "guidance": "高 UV 时采用遮阳、衣物与合规防晒措施。"},
        {"key": "humidity", "label": "湿热与降水", "value": f"{round(humidity)}% / {round(rain)}%", "score": damp_score, "level": damp_level, "icon": "cloud-rain", "guidance": "保持通风干燥，雨后及时更换潮湿衣物。"},
    ]


def _meal_plan(temperature, profile):
    allergies = set(profile.get("allergies", []))
    diets = set(profile.get("dietary_preferences", []))
    conditions = set(profile.get("conditions", []))
    plant_based = bool({"vegetarian", "vegan"} & diets)

    breakfast_id = "berry-millet-porridge" if temperature <= 8 or "gluten" in allergies else "berry-oat-cup"
    lunch_id = "roasted-vegetable-quinoa-bowl" if (plant_based or "seafood" in allergies or "gout" in conditions) else "pan-seared-fish-bowl"
    dinner_id = "tomato-mushroom-soup"

    breakfast_note = "全谷物搭配完整水果，不额外加入糖浆。"
    if breakfast_id == "berry-millet-porridge":
        breakfast_note = "使用无麸质小米与完整水果，温热软糯。"

    lunch_note = "少油煎制，搭配两种以上时蔬和适量全谷物。"
    if lunch_id == "roasted-vegetable-quinoa-bowl":
        lunch_note = "多彩时蔬、藜麦与鹰嘴豆组成植物性均衡餐碗。"

    dinner_note = "晚餐减少油盐，以蔬菜汤配少量杂粮主食。"
    if temperature <= 8:
        dinner_id = "pumpkin-chickpea-soup" if plant_based else "pumpkin-chicken-soup"
        dinner_note = "温热少油，搭配深色蔬菜，避免过烫进食。"
    if "gout" in conditions:
        dinner_id = "winter-melon-egg-soup" if "egg" not in allergies and "vegan" not in diets else "winter-melon-vegetable-soup"
        dinner_note = "减少浓肉汤、动物内脏与酒精，饮水需求按医嘱调整。"

    condition_note = ""
    if "hypertension" in conditions or "low_sodium" in diets:
        condition_note = " 控盐并少用酱料。"
    if "diabetes" in conditions or "low_sugar" in diets:
        condition_note += " 主食定量，不以果汁替代完整水果。"
    if "kidney_disease" in conditions:
        condition_note += " 具体食材份量需按肾功能与化验指标调整。"

    selected = [
        ("早餐", "07:30", breakfast_id, breakfast_note, "berry", "清醒启动"),
        ("午餐", "12:10", lunch_id, lunch_note, "citrus", "稳定供能"),
        ("晚餐", "18:30", dinner_id, dinner_note, "leaf", "轻盈收束"),
    ]
    meals = []
    for slot, meal_time, recipe_id, note, tone, energy in selected:
        recipe = get_recipe_summary(recipe_id, profile)
        meals.append({
            "slot": slot,
            "time": meal_time,
            "recipe_id": recipe_id,
            "title": recipe["title"],
            "description": note + condition_note,
            "image": recipe["image"],
            "image_alt": recipe["image_alt"],
            "tone": tone,
            "energy": energy,
            "fit_score": min(98, recipe["profile_match"]["score"] + 2),
            "duration_minutes": recipe["prep_minutes"] + recipe["cook_minutes"],
            "difficulty": recipe["difficulty"],
            "evidence": [
                {"key": "climate", "label": "今日环境", "icon": "cloud-sun", "detail": _meal_climate_reason(slot, temperature, recipe_id)},
                {"key": "profile", "label": "健康档案", "icon": "shield-check", "detail": _meal_profile_reason(profile, recipe)},
                {"key": "nutrition", "label": "餐次结构", "icon": "utensils", "detail": _meal_nutrition_reason(slot, recipe_id)},
            ],
        })
    return meals


def _meal_climate_reason(slot, temperature, recipe_id):
    rounded = round(temperature)
    if temperature <= 8:
        return f"当前体感约 {rounded}°C，{slot}采用温热、少油并避免过烫的方案。"
    if temperature >= 30:
        return f"当前体感约 {rounded}°C，{slot}减少厚重酱汁，优先清爽补水型食材。"
    if recipe_id == "tomato-mushroom-soup":
        return f"当前体感约 {rounded}°C，晚餐以轻汤收束，避免增加夜间消化负担。"
    return f"当前体感约 {rounded}°C，餐食保持温度适中与稳定供能。"


def _meal_profile_reason(profile, recipe):
    exclusions = _excluded_labels(profile)
    if exclusions:
        return f"已按本机档案过滤{'、'.join(exclusions[:4])}，当前食谱兼容性 {recipe['profile_match']['score']}%。"
    if profile.get("conditions") or profile.get("health_goals"):
        return f"已核对慢性疾病、健康目标与饮食偏好，当前兼容性 {recipe['profile_match']['score']}%。"
    return "当前档案未记录食物禁忌；已按默认成年人均衡饮食规则核对。"


def _meal_nutrition_reason(slot, recipe_id):
    if slot == "早餐":
        return "以全谷物和完整水果建立饱腹感，不使用果汁替代水果。"
    if slot == "午餐":
        return "蔬菜、全谷物与蛋白来源组合，承担今日主要稳定供能。"
    if recipe_id.startswith("winter-melon"):
        return "采用清水汤底，避免浓肉汤、海鲜汤底和高盐复合调料。"
    return "晚餐使用可量化的少油配方，并保留蔬菜体积。"


def _hydration_plan(temperature, profile):
    restricted = bool({"kidney_disease", "cardiovascular"} & set(profile.get("conditions", [])))
    if restricted:
        return {"target": "按医嘱个体化", "amount_ml": None, "note": "基础疾病可能需要限制或调整液体摄入。"}
    amount = 1600
    if temperature >= 30:
        amount += 400
    elif temperature <= 5:
        amount += 100
    return {"target": f"约 {amount} ml", "amount_ml": amount, "note": "少量多次；大量出汗或运动时结合实际补充。"}


def _wellness_timeline(snapshot, aqi, uv, rain):
    sunrise = _clock(snapshot["today"].get("sunrise"), "06:30")
    sunset = _clock(snapshot["today"].get("sunset"), "18:30")
    outdoor = "室内舒展 15 分钟" if (aqi or 0) > 150 or rain >= 70 else "户外轻活动 20 分钟"
    noon = "午间减少暴晒" if uv >= 6 else "午间短时散步"
    return [
        {"time": sunrise, "phase": "唤醒", "title": "晨光启动", "detail": "拉开窗帘，饮水并完成关节活动。", "icon": "sunrise"},
        {"time": "10:30", "phase": "活力", "title": outdoor, "detail": "根据空气质量与降水动态调整场地。", "icon": "footprints"},
        {"time": "12:30", "phase": "防护", "title": noon, "detail": "高温或高 UV 时优先选择遮阴与室内空间。", "icon": "umbrella"},
        {"time": sunset, "phase": "恢复", "title": "降低身体负荷", "detail": "清淡晚餐，睡前减少高强度运动和刺激饮品。", "icon": "moon-star"},
    ]


def _protection_plan(temperature, aqi, uv, humidity, profile):
    items = []
    if temperature >= 30:
        items.append({"title": "热相关不适", "level": "high" if temperature >= 35 else "elevated", "icon": "thermometer-sun", "actions": ["避开午后高温时段", "分次补水并观察头晕乏力", "保持室内通风或合理降温"]})
    elif temperature <= 8:
        items.append({"title": "心脑血管冷刺激", "level": "elevated", "icon": "heart-pulse", "actions": ["起床与外出时逐步保暖", "避免骤然高强度运动", "按医嘱监测血压与症状"]})
    if (aqi or 0) > 100:
        items.append({"title": "呼吸道刺激", "level": "high" if (aqi or 0) > 150 else "elevated", "icon": "wind", "actions": ["缩短户外停留时间", "关闭污染侧窗户并适时通风", "出现喘憋或胸痛及时就医"]})
    if uv >= 6:
        items.append({"title": "紫外线损伤", "level": "high" if uv >= 8 else "elevated", "icon": "sun", "actions": ["优先使用衣物与遮阳工具", "暴露部位使用合规防晒品", "避免长时间正午暴晒"]})
    if humidity >= 75:
        items.append({"title": "潮湿环境风险", "level": "guarded", "icon": "droplets", "actions": ["保持居室干燥通风", "清理积水与霉变区域", "皮肤与衣物保持清洁干燥"]})
    if not items:
        items.append({"title": "常规健康维护", "level": "low", "icon": "shield-check", "actions": ["保持规律作息", "完成适量身体活动", "均衡饮食并关注身体信号"]})
    return items[:4]


def _recommendation_cache_key(snapshot, profile):
    current = snapshot["current"]
    today = snapshot["today"]
    fingerprint = {
        "date": date.today().isoformat(),
        "location": [round(snapshot["location"]["latitude"], 2), round(snapshot["location"]["longitude"], 2)],
        "weather": [
            current.get("weather_group"),
            _band(current.get("feels_like"), (0, 8, 18, 30, 35)),
            _band(current.get("aqi"), (50, 100, 150, 200)),
            _band(today.get("uv_index_max"), (3, 6, 8, 11)),
            _band(today.get("precipitation_probability_max"), (30, 70)),
        ],
        "profile": profile,
        "rules": "2026.07.3",
    }
    raw = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validated_choices(value, allowed, label):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ProfileValidationError(f"{label}必须是数组")
    result = list(dict.fromkeys(str(item) for item in value))
    if not set(result) <= allowed:
        raise ProfileValidationError(f"{label}包含无效选项")
    return result


def _json_list(value):
    try:
        result = json.loads(value or "[]")
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        return []


def _band(value, thresholds):
    value = value if value is not None else -999
    return sum(value >= threshold for threshold in thresholds)


def _clock(value, fallback):
    return str(value).split("T", 1)[1][:5] if value and "T" in str(value) else fallback


def _excluded_labels(profile):
    labels = {
        "dairy": "乳制品", "egg": "蛋类", "nuts": "坚果", "seafood": "海鲜",
        "soy": "大豆", "gluten": "麸质", "vegetarian": "素食", "vegan": "纯素",
        "low_sodium": "低盐", "low_sugar": "低糖",
    }
    keys = profile.get("allergies", []) + profile.get("dietary_preferences", [])
    return [labels[key] for key in keys if key in labels]
