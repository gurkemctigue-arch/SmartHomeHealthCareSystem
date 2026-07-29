"""Business rules for VITAL QUEST progression, editorial content and game scoring."""

from __future__ import annotations

from datetime import datetime
import json
import random
import re
import uuid

from flask import session

from database.db import get_db
from services.quest_content import GAME_ACTIONS, GAME_CASES


LEVELS = [
    (0, "健康观察员"),
    (80, "知识调查员"),
    (200, "安全用药官"),
    (400, "社区守护者"),
    (700, "生命领航员"),
]

EVENT_LABELS = {
    "article_read": "完成知识阅读",
    "article_quiz": "通过知识校验",
    "article_quiz_perfect": "测验获得满分",
    "comment": "发布有效讨论",
    "game_shift": "完成急诊班次",
    "achievement": "解锁成就",
}

REVIEW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"确诊为", r"保证治愈", r"立即停药", r"每天\s*\d+\s*(片|粒|毫克|mg)",
        r"不用看医生", r"替代医生", r"包治", r"神药",
    )
]


class QuestValidationError(ValueError):
    """Raised for malformed or unsafe VITAL QUEST requests."""


class QuestNotFoundError(LookupError):
    """Raised when a requested quest resource does not exist."""


def current_user_id():
    db = get_db()
    user_id = session.get("quest_user_id", 1)
    row = db.execute("SELECT id FROM quest_user WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        row = db.execute("SELECT id FROM quest_user ORDER BY id LIMIT 1").fetchone()
    if row is None:
        raise QuestNotFoundError("没有可用的探索者档案")
    session["quest_user_id"] = row["id"]
    return row["id"]


def list_profiles():
    active_id = current_user_id()
    rows = get_db().execute(
        "SELECT id, username, display_name, avatar_seed, created_at FROM quest_user ORDER BY id"
    ).fetchall()
    return [{**dict(row), "active": row["id"] == active_id} for row in rows]


def create_profile(payload):
    if not isinstance(payload, dict):
        raise QuestValidationError("档案内容必须是 JSON 对象")
    display_name = str(payload.get("display_name") or "").strip()
    if not 2 <= len(display_name) <= 24:
        raise QuestValidationError("显示名称需要 2 到 24 个字符")
    username = re.sub(r"[^a-z0-9_-]", "", str(payload.get("username") or "").lower())
    if not username:
        username = f"explorer-{uuid.uuid4().hex[:8]}"
    if not 3 <= len(username) <= 32:
        raise QuestValidationError("用户名需要 3 到 32 个英文字符")
    avatar_seed = str(payload.get("avatar_seed") or "aurora")[:24]
    db = get_db()
    try:
        cursor = db.execute(
            "INSERT INTO quest_user (username, display_name, avatar_seed) VALUES (?, ?, ?)",
            (username, display_name, avatar_seed),
        )
        db.commit()
    except Exception as exc:
        raise QuestValidationError("用户名已经存在") from exc
    session["quest_user_id"] = cursor.lastrowid
    return _profile(cursor.lastrowid)


def activate_profile(user_id):
    row = get_db().execute("SELECT id FROM quest_user WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise QuestNotFoundError("探索者档案不存在")
    session["quest_user_id"] = row["id"]
    return _profile(row["id"])


def get_quest_overview():
    user_id = current_user_id()
    _evaluate_achievements(user_id)
    db = get_db()
    profile = _profile(user_id)
    achievements = _achievement_rows(user_id)

    completed_articles = db.execute(
        "SELECT COUNT(*) FROM quest_article_progress WHERE user_id = ? AND completed_at IS NOT NULL",
        (user_id,),
    ).fetchone()[0]
    total_articles = db.execute(
        "SELECT COUNT(*) FROM quest_article WHERE status = 'published'"
    ).fetchone()[0]
    game_stats = db.execute(
        "SELECT COUNT(*) AS shifts, COALESCE(SUM(correct_actions), 0) AS correct, "
        "COALESCE(MAX(score), 0) AS best_score FROM quest_game_session "
        "WHERE user_id = ? AND status = 'completed'",
        (user_id,),
    ).fetchone()
    comments = db.execute(
        "SELECT COUNT(*) FROM quest_comment WHERE user_id = ? AND status = 'published'",
        (user_id,),
    ).fetchone()[0]

    weekly = {
        "articles": _weekly_count(
            "quest_article_progress", "user_id = ? AND completed_at IS NOT NULL", "completed_at", user_id
        ),
        "shifts": _weekly_count(
            "quest_game_session", "user_id = ? AND status = 'completed'", "completed_at", user_id
        ),
        "comments": _weekly_count(
            "quest_comment", "user_id = ? AND status = 'published'", "created_at", user_id
        ),
    }
    missions = [
        {"key": "articles", "label": "完成 3 篇知识阅读", "current": min(weekly["articles"], 3), "target": 3, "icon": "book-open-check"},
        {"key": "shifts", "label": "完成 2 个安全班次", "current": min(weekly["shifts"], 2), "target": 2, "icon": "siren"},
        {"key": "comments", "label": "参与 2 次有效讨论", "current": min(weekly["comments"], 2), "target": 2, "icon": "messages-square"},
    ]

    recent = db.execute(
        "SELECT event_type, reference_type, reference_id, points, created_at "
        "FROM quest_points_ledger WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 8",
        (user_id,),
    ).fetchall()

    return {
        "profile": profile,
        "stats": {
            "articles_completed": completed_articles,
            "articles_total": total_articles,
            "game_shifts": game_stats["shifts"],
            "correct_actions": game_stats["correct"],
            "best_score": game_stats["best_score"],
            "comments": comments,
        },
        "missions": missions,
        "achievements": achievements,
        "recent_activity": [
            {
                **dict(row),
                "label": EVENT_LABELS.get(row["event_type"], "获得 Vital Points"),
            }
            for row in recent
        ],
    }


def list_articles(category="all", query=""):
    user_id = current_user_id()
    clauses = ["a.status = 'published'"]
    values = [user_id]
    if category and category != "all":
        clauses.append("a.category = ?")
        values.append(str(category))
    query = str(query or "").strip()
    if query:
        clauses.append("(a.title LIKE ? OR a.dek LIKE ? OR a.category LIKE ?)")
        term = f"%{query[:60]}%"
        values.extend([term, term, term])
    rows = get_db().execute(
        "SELECT a.id, a.slug, a.title, a.dek, a.category, a.source_name, a.reviewer, "
        "a.reviewed_at, a.reading_minutes, a.cover_asset, a.accent, "
        "COALESCE(p.progress, 0) AS progress, COALESCE(p.best_quiz_score, 0) AS best_quiz_score, "
        "p.completed_at, (SELECT COUNT(*) FROM quest_comment c "
        "WHERE c.article_id = a.id AND c.status = 'published') AS comment_count "
        "FROM quest_article a LEFT JOIN quest_article_progress p "
        "ON p.article_id = a.id AND p.user_id = ? "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY CASE WHEN p.completed_at IS NULL THEN 0 ELSE 1 END, a.reviewed_at DESC, a.id",
        values,
    ).fetchall()
    categories = [
        row[0] for row in get_db().execute(
            "SELECT DISTINCT category FROM quest_article WHERE status = 'published' ORDER BY category"
        ).fetchall()
    ]
    return {"items": [dict(row) for row in rows], "categories": categories}


def get_article(slug):
    user_id = current_user_id()
    row = get_db().execute(
        "SELECT a.*, COALESCE(p.progress, 0) AS progress, "
        "COALESCE(p.best_quiz_score, 0) AS best_quiz_score, p.completed_at "
        "FROM quest_article a LEFT JOIN quest_article_progress p "
        "ON p.article_id = a.id AND p.user_id = ? "
        "WHERE a.slug = ? AND a.status = 'published'",
        (user_id, slug),
    ).fetchone()
    if row is None:
        raise QuestNotFoundError("文章不存在")
    item = dict(row)
    item["body"] = _json(item["body"], [])
    item["takeaways"] = _json(item["takeaways"], [])
    quiz = _json(item.pop("quiz"), [])
    item["quiz"] = [
        {"index": index, "question": question["question"], "options": question["options"]}
        for index, question in enumerate(quiz)
    ]
    item["comments"] = get_comments(item["id"])
    return item


def update_article_progress(slug, progress):
    user_id = current_user_id()
    article = _article_row(slug)
    try:
        progress = max(0, min(100, int(progress)))
    except (TypeError, ValueError) as exc:
        raise QuestValidationError("阅读进度必须是 0 到 100 的整数") from exc
    db = get_db()
    existing = db.execute(
        "SELECT progress, completed_at FROM quest_article_progress WHERE user_id = ? AND article_id = ?",
        (user_id, article["id"]),
    ).fetchone()
    best_progress = max(progress, existing["progress"] if existing else 0)
    completed_now = best_progress >= 90 and (existing is None or existing["completed_at"] is None)
    db.execute(
        "INSERT INTO quest_article_progress (user_id, article_id, progress, completed_at, last_read_at) "
        "VALUES (?, ?, ?, CASE WHEN ? >= 90 THEN datetime('now', 'localtime') ELSE NULL END, "
        "datetime('now', 'localtime')) ON CONFLICT(user_id, article_id) DO UPDATE SET "
        "progress=MAX(progress, excluded.progress), completed_at=CASE "
        "WHEN quest_article_progress.completed_at IS NULL AND excluded.progress >= 90 "
        "THEN datetime('now', 'localtime') ELSE quest_article_progress.completed_at END, "
        "last_read_at=excluded.last_read_at",
        (user_id, article["id"], best_progress, best_progress),
    )
    awarded = 0
    if completed_now:
        awarded = _award_points(
            user_id,
            f"article-read:{user_id}:{article['id']}",
            "article_read",
            "article",
            str(article["id"]),
            5,
            {"title": article["title"]},
        )
    db.commit()
    unlocked = _evaluate_achievements(user_id)
    return {"progress": best_progress, "completed": best_progress >= 90, "points_awarded": awarded, "unlocked": unlocked}


def submit_article_quiz(slug, answers):
    user_id = current_user_id()
    article = _article_row(slug)
    if not isinstance(answers, list):
        raise QuestValidationError("答案必须是数组")
    quiz = _json(article["quiz"], [])
    if len(answers) != len(quiz):
        raise QuestValidationError("请完成全部题目")
    normalized = []
    for answer in answers:
        try:
            normalized.append(int(answer))
        except (TypeError, ValueError) as exc:
            raise QuestValidationError("答案格式不正确") from exc
    correctness = [answer == question["answer"] for answer, question in zip(normalized, quiz)]
    correct_count = sum(correctness)
    score = round(correct_count / max(len(quiz), 1) * 100)
    db = get_db()
    db.execute(
        "INSERT INTO quest_article_progress (user_id, article_id, progress, best_quiz_score, last_read_at) "
        "VALUES (?, ?, 0, ?, datetime('now', 'localtime')) "
        "ON CONFLICT(user_id, article_id) DO UPDATE SET "
        "best_quiz_score=MAX(best_quiz_score, excluded.best_quiz_score), last_read_at=excluded.last_read_at",
        (user_id, article["id"], score),
    )
    points = 0
    if score >= 67:
        points += _award_points(
            user_id, f"article-quiz:{user_id}:{article['id']}", "article_quiz",
            "article", str(article["id"]), 10, {"score": score, "title": article["title"]},
        )
    if score == 100:
        points += _award_points(
            user_id, f"article-perfect:{user_id}:{article['id']}", "article_quiz_perfect",
            "article", str(article["id"]), 5, {"score": score, "title": article["title"]},
        )
    db.commit()
    unlocked = _evaluate_achievements(user_id)
    return {
        "score": score,
        "correct_count": correct_count,
        "total": len(quiz),
        "correctness": correctness,
        "points_awarded": points,
        "unlocked": unlocked,
    }


def get_comments(article_id):
    user_id = current_user_id()
    rows = get_db().execute(
        "SELECT c.id, c.content, c.status, c.helpful_count, c.created_at, "
        "u.display_name, u.avatar_seed, CASE WHEN c.user_id = ? THEN 1 ELSE 0 END AS mine "
        "FROM quest_comment c JOIN quest_user u ON u.id = c.user_id "
        "WHERE c.article_id = ? AND (c.status = 'published' OR c.user_id = ?) "
        "ORDER BY c.created_at DESC, c.id DESC LIMIT 50",
        (user_id, article_id, user_id),
    ).fetchall()
    return [dict(row) for row in rows]


def add_comment(slug, content):
    user_id = current_user_id()
    article = _article_row(slug)
    content = " ".join(str(content or "").strip().split())
    if not 8 <= len(content) <= 500:
        raise QuestValidationError("评论需要 8 到 500 个字符")
    status = "review" if any(pattern.search(content) for pattern in REVIEW_PATTERNS) else "published"
    db = get_db()
    cursor = db.execute(
        "INSERT INTO quest_comment (user_id, article_id, content, status) VALUES (?, ?, ?, ?)",
        (user_id, article["id"], content, status),
    )
    points = 0
    if status == "published":
        rewarded_today = db.execute(
            "SELECT COUNT(*) FROM quest_points_ledger WHERE user_id = ? AND event_type = 'comment' "
            "AND date(created_at) = date('now', 'localtime')",
            (user_id,),
        ).fetchone()[0]
        if rewarded_today < 3:
            points = _award_points(
                user_id, f"comment:{user_id}:{cursor.lastrowid}", "comment", "comment",
                str(cursor.lastrowid), 2, {"article": article["title"]},
            )
    db.commit()
    unlocked = _evaluate_achievements(user_id)
    return {
        "id": cursor.lastrowid,
        "status": status,
        "points_awarded": points,
        "message": "评论已发布" if status == "published" else "评论包含具体医疗建议，已进入人工复核",
        "unlocked": unlocked,
    }


def start_game_session():
    user_id = current_user_id()
    public_id = uuid.uuid4().hex
    case_ids = [case["id"] for case in GAME_CASES]
    random.SystemRandom().shuffle(case_ids)
    case_ids = case_ids[:8]
    state = {"queue": case_ids, "results": []}
    db = get_db()
    db.execute(
        "INSERT INTO quest_game_session (public_id, user_id, scenario_key, event_log) "
        "VALUES (?, ?, 'night-shift-01', ?)",
        (public_id, user_id, json.dumps(state, ensure_ascii=False)),
    )
    db.commit()
    cases = [_public_case(_case_by_id(case_id)) for case_id in case_ids]
    return {
        "session_id": public_id,
        "scenario": {
            "key": "night-shift-01",
            "name": "01 / 夜间交接班",
            "duration_seconds": 120,
            "brief": "核对医嘱、识别红线，在速度和安全之间做出正确选择。",
        },
        "actions": GAME_ACTIONS,
        "cases": cases,
    }


def submit_game_action(public_id, payload):
    user_id = current_user_id()
    if not isinstance(payload, dict):
        raise QuestValidationError("操作内容必须是 JSON 对象")
    case_id = str(payload.get("case_id") or "")
    action = str(payload.get("action") or "")
    if action not in {item["key"] for item in GAME_ACTIONS}:
        raise QuestValidationError("未知处置动作")
    db = get_db()
    game = db.execute(
        "SELECT * FROM quest_game_session WHERE public_id = ? AND user_id = ?",
        (public_id, user_id),
    ).fetchone()
    if game is None:
        raise QuestNotFoundError("游戏班次不存在")
    if game["status"] != "active":
        raise QuestValidationError("这个班次已经结束")
    state = _json(game["event_log"], {"queue": [], "results": []})
    if case_id not in state.get("queue", []):
        raise QuestValidationError("病例不属于当前班次")
    if any(item["case_id"] == case_id for item in state.get("results", [])):
        raise QuestValidationError("该患者已经完成处置")
    case = _case_by_id(case_id)
    correct = action == case["correct_action"]
    unsafe = not correct and action in case.get("unsafe_actions", [])
    previous_results = state.get("results", [])
    combo = 1
    for item in reversed(previous_results):
        if not item["correct"]:
            break
        combo += 1
    delta = 100 + min(combo - 1, 4) * 10 if correct else (-45 if unsafe else -15)
    result = {
        "case_id": case_id,
        "action": action,
        "correct": correct,
        "unsafe": unsafe,
        "delta": delta,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    state.setdefault("results", []).append(result)
    score = max(0, int(game["score"]) + delta)
    patients = len(state["results"])
    correct_actions = sum(1 for item in state["results"] if item["correct"])
    unsafe_actions = sum(1 for item in state["results"] if item["unsafe"])
    safety = max(0, 100 - unsafe_actions * 20)
    db.execute(
        "UPDATE quest_game_session SET event_log = ?, score = ?, safety_score = ?, "
        "patients_served = ?, correct_actions = ?, unsafe_actions = ? WHERE id = ?",
        (
            json.dumps(state, ensure_ascii=False), score, safety, patients,
            correct_actions, unsafe_actions, game["id"],
        ),
    )
    db.commit()
    correct_meta = next(item for item in GAME_ACTIONS if item["key"] == case["correct_action"])
    return {
        "correct": correct,
        "unsafe": unsafe,
        "score_delta": delta,
        "score": score,
        "safety_score": safety,
        "combo": combo if correct else 0,
        "correct_action": correct_meta,
        "explanation": case["explanation"],
        "article_slug": case["article_slug"],
        "completed_cases": patients,
        "total_cases": len(state["queue"]),
    }


def finish_game_session(public_id):
    user_id = current_user_id()
    db = get_db()
    game = db.execute(
        "SELECT * FROM quest_game_session WHERE public_id = ? AND user_id = ?",
        (public_id, user_id),
    ).fetchone()
    if game is None:
        raise QuestNotFoundError("游戏班次不存在")
    if game["status"] == "completed":
        return _game_summary(game, 0, [])
    if game["patients_served"] < 1:
        raise QuestValidationError("至少完成一位患者的处置")
    started = datetime.fromisoformat(game["started_at"])
    duration = max(1, round((datetime.now() - started).total_seconds()))
    safety_bonus = 100 if game["unsafe_actions"] == 0 else 0
    final_score = int(game["score"]) + safety_bonus
    db.execute(
        "UPDATE quest_game_session SET status = 'completed', score = ?, duration_seconds = ?, "
        "completed_at = datetime('now', 'localtime') WHERE id = ?",
        (final_score, duration, game["id"]),
    )
    points = min(80, max(10, int(game["correct_actions"]) * 6 + (10 if game["unsafe_actions"] == 0 else 0)))
    awarded = _award_points(
        user_id, f"game-shift:{user_id}:{game['public_id']}", "game_shift", "game_session",
        str(game["id"]), points,
        {"score": final_score, "safety": game["safety_score"], "patients": game["patients_served"]},
    )
    db.commit()
    unlocked = _evaluate_achievements(user_id)
    refreshed = db.execute("SELECT * FROM quest_game_session WHERE id = ?", (game["id"],)).fetchone()
    return _game_summary(refreshed, awarded, unlocked)


def _profile(user_id):
    db = get_db()
    row = db.execute(
        "SELECT id, username, display_name, avatar_seed, created_at FROM quest_user WHERE id = ?",
        (user_id,),
    ).fetchone()
    if row is None:
        raise QuestNotFoundError("探索者档案不存在")
    xp = db.execute(
        "SELECT COALESCE(SUM(points), 0) FROM quest_points_ledger WHERE user_id = ?",
        (user_id,),
    ).fetchone()[0]
    level_index = max(index for index, (threshold, _name) in enumerate(LEVELS) if xp >= threshold)
    threshold, level_name = LEVELS[level_index]
    if level_index + 1 < len(LEVELS):
        next_threshold, next_name = LEVELS[level_index + 1]
        level_progress = round((xp - threshold) / (next_threshold - threshold) * 100)
    else:
        next_threshold, next_name, level_progress = threshold, level_name, 100
    streak = db.execute(
        "SELECT COUNT(DISTINCT date(created_at)) FROM quest_points_ledger "
        "WHERE user_id = ? AND created_at >= datetime('now', '-6 days', 'localtime')",
        (user_id,),
    ).fetchone()[0]
    return {
        **dict(row),
        "xp": xp,
        "level": level_index + 1,
        "level_name": level_name,
        "next_level_name": next_name,
        "next_level_xp": next_threshold,
        "level_progress": min(100, max(0, level_progress)),
        "active_days": streak,
    }


def _achievement_rows(user_id):
    rows = get_db().execute(
        "SELECT a.achievement_key, a.name, a.description, a.icon, a.tone, a.bonus_points, "
        "ua.unlocked_at FROM quest_achievement a LEFT JOIN quest_user_achievement ua "
        "ON ua.achievement_id = a.id AND ua.user_id = ? ORDER BY a.id",
        (user_id,),
    ).fetchall()
    return [{**dict(row), "unlocked": row["unlocked_at"] is not None} for row in rows]


def _evaluate_achievements(user_id):
    db = get_db()
    stats = {
        "articles": db.execute(
            "SELECT COUNT(*) FROM quest_article_progress WHERE user_id = ? AND completed_at IS NOT NULL",
            (user_id,),
        ).fetchone()[0],
        "perfect": db.execute(
            "SELECT COUNT(*) FROM quest_article_progress WHERE user_id = ? AND best_quiz_score = 100",
            (user_id,),
        ).fetchone()[0],
        "comments": db.execute(
            "SELECT COUNT(*) FROM quest_comment WHERE user_id = ? AND status = 'published'",
            (user_id,),
        ).fetchone()[0],
        "shifts": db.execute(
            "SELECT COUNT(*) FROM quest_game_session WHERE user_id = ? AND status = 'completed'",
            (user_id,),
        ).fetchone()[0],
        "safe_shifts": db.execute(
            "SELECT COUNT(*) FROM quest_game_session WHERE user_id = ? AND status = 'completed' "
            "AND patients_served > 0 AND unsafe_actions = 0",
            (user_id,),
        ).fetchone()[0],
        "correct": db.execute(
            "SELECT COALESCE(SUM(correct_actions), 0) FROM quest_game_session "
            "WHERE user_id = ? AND status = 'completed'",
            (user_id,),
        ).fetchone()[0],
        "xp": db.execute(
            "SELECT COALESCE(SUM(points), 0) FROM quest_points_ledger WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0],
    }
    conditions = {
        "first_signal": stats["articles"] >= 1,
        "perfect_recall": stats["perfect"] >= 1,
        "thoughtful_voice": stats["comments"] >= 3,
        "first_shift": stats["shifts"] >= 1,
        "safety_first": stats["safe_shifts"] >= 1,
        "triage_operator": stats["correct"] >= 12,
        "knowledge_path": stats["articles"] >= 5,
        "vital_guardian": stats["xp"] >= 500,
    }
    unlocked = []
    for key, ready in conditions.items():
        if not ready:
            continue
        achievement = db.execute(
            "SELECT * FROM quest_achievement WHERE achievement_key = ?", (key,)
        ).fetchone()
        if achievement is None:
            continue
        cursor = db.execute(
            "INSERT OR IGNORE INTO quest_user_achievement (user_id, achievement_id) VALUES (?, ?)",
            (user_id, achievement["id"]),
        )
        if cursor.rowcount:
            unlocked.append({
                "key": key,
                "name": achievement["name"],
                "description": achievement["description"],
                "icon": achievement["icon"],
                "tone": achievement["tone"],
            })
            if achievement["bonus_points"]:
                _award_points(
                    user_id, f"achievement:{user_id}:{key}", "achievement", "achievement",
                    key, achievement["bonus_points"], {"name": achievement["name"]},
                )
    db.commit()
    return unlocked


def _award_points(user_id, event_key, event_type, reference_type, reference_id, points, metadata):
    cursor = get_db().execute(
        "INSERT OR IGNORE INTO quest_points_ledger "
        "(user_id, event_key, event_type, reference_type, reference_id, points, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            user_id, event_key, event_type, reference_type, reference_id, int(points),
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )
    return int(points) if cursor.rowcount else 0


def _weekly_count(table, condition, date_column, user_id):
    return get_db().execute(
        f"SELECT COUNT(*) FROM {table} WHERE {condition} "
        f"AND {date_column} >= datetime('now', '-6 days', 'localtime')",
        (user_id,),
    ).fetchone()[0]


def _article_row(slug):
    row = get_db().execute(
        "SELECT * FROM quest_article WHERE slug = ? AND status = 'published'", (slug,)
    ).fetchone()
    if row is None:
        raise QuestNotFoundError("文章不存在")
    return row


def _case_by_id(case_id):
    case = next((item for item in GAME_CASES if item["id"] == case_id), None)
    if case is None:
        raise QuestNotFoundError("病例不存在")
    return case


def _public_case(case):
    return {
        key: case[key]
        for key in ("id", "patient", "age", "complaint", "brief", "tag")
    }


def _game_summary(game, points_awarded, unlocked):
    return {
        "session_id": game["public_id"],
        "score": game["score"],
        "safety_score": game["safety_score"],
        "patients_served": game["patients_served"],
        "correct_actions": game["correct_actions"],
        "unsafe_actions": game["unsafe_actions"],
        "duration_seconds": game["duration_seconds"],
        "points_awarded": points_awarded,
        "unlocked": unlocked,
    }


def _json(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback
