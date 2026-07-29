"""Flask blueprint for the VITAL QUEST achievement and learning zone."""

from flask import Blueprint, jsonify, request

from services.quest_service import (
    QuestNotFoundError,
    QuestValidationError,
    activate_profile,
    add_comment,
    create_profile,
    finish_game_session,
    get_article,
    get_quest_overview,
    list_articles,
    list_profiles,
    start_game_session,
    submit_article_quiz,
    submit_game_action,
    update_article_progress,
)


quest_bp = Blueprint("quest", __name__, url_prefix="/api/quest")


def _ok(data, message="success", code=200):
    return jsonify({"code": code, "message": message, "data": data}), code


@quest_bp.get("/overview")
def quest_overview():
    return _ok(get_quest_overview())


@quest_bp.get("/profiles")
def quest_profiles():
    return _ok(list_profiles())


@quest_bp.post("/profiles")
def quest_profile_create():
    return _ok(create_profile(request.get_json(silent=True)), "created", 201)


@quest_bp.post("/profiles/<int:user_id>/activate")
def quest_profile_activate(user_id):
    return _ok(activate_profile(user_id), "activated")


@quest_bp.get("/articles")
def quest_articles():
    return _ok(list_articles(request.args.get("category", "all"), request.args.get("q", "")))


@quest_bp.get("/articles/<slug>")
def quest_article_detail(slug):
    return _ok(get_article(slug))


@quest_bp.post("/articles/<slug>/progress")
def quest_article_progress(slug):
    body = request.get_json(silent=True) or {}
    return _ok(update_article_progress(slug, body.get("progress")), "updated")


@quest_bp.post("/articles/<slug>/quiz")
def quest_article_quiz(slug):
    body = request.get_json(silent=True) or {}
    return _ok(submit_article_quiz(slug, body.get("answers")), "evaluated")


@quest_bp.post("/articles/<slug>/comments")
def quest_article_comment(slug):
    body = request.get_json(silent=True) or {}
    return _ok(add_comment(slug, body.get("content")), "created", 201)


@quest_bp.post("/game/sessions")
def quest_game_start():
    return _ok(start_game_session(), "started", 201)


@quest_bp.post("/game/sessions/<public_id>/action")
def quest_game_action(public_id):
    return _ok(submit_game_action(public_id, request.get_json(silent=True)), "evaluated")


@quest_bp.post("/game/sessions/<public_id>/finish")
def quest_game_finish(public_id):
    return _ok(finish_game_session(public_id), "completed")


@quest_bp.errorhandler(QuestValidationError)
def quest_validation_error(error):
    return jsonify({"code": 400, "message": str(error), "data": None}), 400


@quest_bp.errorhandler(QuestNotFoundError)
def quest_not_found(error):
    return jsonify({"code": 404, "message": str(error), "data": None}), 404
