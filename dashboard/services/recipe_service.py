"""Curated recipe catalog and deterministic health-profile compatibility checks."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path


CATALOG_PATH = Path(__file__).resolve().parents[1] / "data" / "wellness_recipes.json"

ALLERGEN_LABELS = {
    "dairy": "乳制品",
    "egg": "蛋类",
    "nuts": "坚果",
    "seafood": "海鲜",
    "soy": "大豆",
    "gluten": "麸质",
}

CONDITION_LABELS = {
    "hypertension": "高血压",
    "diabetes": "糖尿病",
    "hyperlipidemia": "高血脂",
    "gout": "痛风",
    "kidney_disease": "肾脏疾病",
    "cardiovascular": "心血管疾病",
    "asthma": "哮喘",
    "allergic_rhinitis": "过敏性鼻炎",
}


class RecipeNotFoundError(LookupError):
    pass


@lru_cache(maxsize=1)
def _catalog():
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise RuntimeError("食谱目录必须是 JSON 数组")

    catalog = {}
    required = {
        "id", "title", "slot", "subtitle", "theme", "image", "servings",
        "nutrition", "ingredients", "steps", "substitutions", "source",
    }
    for recipe in records:
        missing = required - set(recipe)
        if missing:
            raise RuntimeError(f"食谱 {recipe.get('id', '<unknown>')} 缺少字段: {sorted(missing)}")
        if recipe["id"] in catalog:
            raise RuntimeError(f"食谱 ID 重复: {recipe['id']}")
        if len(recipe["steps"]) < 6:
            raise RuntimeError(f"食谱 {recipe['id']} 至少需要 6 个步骤")
        catalog[recipe["id"]] = recipe
    return catalog


def get_recipe(recipe_id):
    recipe = _catalog().get(str(recipe_id or ""))
    if recipe is None:
        raise RecipeNotFoundError("食谱不存在")
    return deepcopy(recipe)


def get_recipe_detail(recipe_id, profile):
    recipe = get_recipe(recipe_id)
    recipe["profile_match"] = evaluate_recipe(recipe, profile or {})
    recipe["nutrition_note"] = "营养数值为每份估算值，会随品牌、产地和实际用量变化。"
    recipe["medical_notice"] = "内容用于日常饮食管理，不替代医生、药师或注册营养师的个体化建议。"
    return recipe


def get_recipe_summary(recipe_id, profile):
    recipe = get_recipe(recipe_id)
    match = evaluate_recipe(recipe, profile or {})
    return {
        "recipe_id": recipe["id"],
        "title": recipe["title"],
        "image": recipe["image"],
        "image_alt": recipe["image_alt"],
        "prep_minutes": recipe["prep_minutes"],
        "cook_minutes": recipe["cook_minutes"],
        "difficulty": recipe["difficulty"],
        "allergens": recipe.get("allergens", []),
        "profile_match": match,
    }


def evaluate_recipe(recipe, profile):
    allergies = set(profile.get("allergies", []))
    diets = set(profile.get("dietary_preferences", []))
    conditions = set(profile.get("conditions", []))
    recipe_allergens = set(recipe.get("allergens", []))
    tags = set(recipe.get("tags", []))

    blockers = []
    adjustments = []
    warnings = []

    for allergen in sorted(allergies & recipe_allergens):
        blockers.append(f"食谱含有已记录的{ALLERGEN_LABELS.get(allergen, allergen)}过敏原")

    if "vegan" in diets and "vegan" not in tags:
        blockers.append("当前档案为纯素饮食，本食谱含动物性食材")
    elif "vegetarian" in diets and not ({"vegetarian", "vegan"} & tags):
        blockers.append("当前档案为素食，本食谱含肉类或水产")

    for condition in sorted(conditions & set(recipe.get("avoid_conditions", []))):
        blockers.append(f"本食谱不适用于已记录的{CONDITION_LABELS.get(condition, condition)}筛选规则")

    nutrition = recipe.get("nutrition", {})
    if "hypertension" in conditions or "low_sodium" in diets:
        adjustments.append("采用无盐汤底，不使用酱油、浓汤宝或复合酱料，盐量以医嘱为准。")
        if nutrition.get("sodium_mg", 0) > 450:
            warnings.append("该食谱每份估算钠偏高，需要进一步减盐。")
    if "diabetes" in conditions or "low_sugar" in diets:
        adjustments.append("谷物、南瓜、玉米和水果均计入本餐碳水总量，不以果汁替代完整水果。")
    if "hyperlipidemia" in conditions or "weight_control" in set(profile.get("health_goals", [])):
        adjustments.append("烹调油按量使用，不额外加入奶油或油炸配料。")
    if "kidney_disease" in conditions:
        warnings.append("肾脏疾病需根据肾功能、血钾、血磷及医嘱重新确认食材和份量。")
    if "cardiovascular" in conditions:
        warnings.append("如有液体或钠摄入限制，以当前医嘱为准。")
    if profile.get("medications"):
        warnings.append("档案已记录用药；食物与药物相互作用需向医生或药师核实。")
    if profile.get("age_group") == "child":
        adjustments.append("儿童食用时切小食材、确认无刺无核，并由照护者调整份量。")
    elif profile.get("age_group") == "older_adult":
        adjustments.append("根据咀嚼和吞咽情况调整软硬度，入口前确认温度。")

    score = max(12, min(99, 96 - len(blockers) * 34 - len(warnings) * 4))
    if blockers:
        status = "ineligible"
        label = "不适合当前档案"
    elif warnings:
        status = "caution"
        label = "需个体化确认"
    else:
        status = "eligible"
        label = "符合当前档案"

    return {
        "eligible": not blockers,
        "status": status,
        "label": label,
        "score": score,
        "blockers": blockers,
        "adjustments": list(dict.fromkeys(adjustments)),
        "warnings": list(dict.fromkeys(warnings)),
        "checked_fields": ["食物过敏", "饮食偏好", "慢性疾病", "健康目标", "年龄段", "用药提示"],
    }


def list_recipe_ids():
    return sorted(_catalog())
