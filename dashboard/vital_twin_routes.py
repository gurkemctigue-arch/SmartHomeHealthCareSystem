"""Flask routes for the family Vital Twin workspace."""

from flask import Blueprint, jsonify, request

from services.vital_twin_service import (
    VitalTwinNotFoundError,
    VitalTwinValidationError,
    add_measurement,
    complete_task,
    create_member,
    create_task,
    delete_member,
    get_vital_twin_overview,
    list_members,
    update_member,
)


vital_twin_bp = Blueprint("vital_twin", __name__, url_prefix="/api/vital-twin")


def _ok(data, message="success", code=200):
    return jsonify({"code": code, "message": message, "data": data}), code


@vital_twin_bp.get("/overview")
def vital_twin_overview():
    return _ok(get_vital_twin_overview(request.args.get("member_id")))


@vital_twin_bp.get("/members")
def vital_twin_members():
    return _ok(list_members())


@vital_twin_bp.post("/members")
def vital_twin_member_create():
    return _ok(create_member(request.get_json(silent=True)), "created", 201)


@vital_twin_bp.put("/members/<int:member_id>")
def vital_twin_member_update(member_id):
    return _ok(update_member(member_id, request.get_json(silent=True)), "updated")


@vital_twin_bp.delete("/members/<int:member_id>")
def vital_twin_member_delete(member_id):
    return _ok(delete_member(member_id), "deleted")


@vital_twin_bp.post("/measurements")
def vital_twin_measurement_create():
    return _ok(add_measurement(request.get_json(silent=True)), "created", 201)


@vital_twin_bp.post("/tasks")
def vital_twin_task_create():
    return _ok(create_task(request.get_json(silent=True)), "created", 201)


@vital_twin_bp.post("/tasks/<int:task_id>/complete")
def vital_twin_task_complete(task_id):
    return _ok(complete_task(task_id), "completed")


@vital_twin_bp.errorhandler(VitalTwinValidationError)
def vital_twin_validation_error(error):
    return jsonify({"code": 400, "message": str(error), "data": None}), 400


@vital_twin_bp.errorhandler(VitalTwinNotFoundError)
def vital_twin_not_found(error):
    return jsonify({"code": 404, "message": str(error), "data": None}), 404
