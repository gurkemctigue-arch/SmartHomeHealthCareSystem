"""药品管理服务 — 从 SQLite 数据库查询"""

import logging
from database.db import get_db

logger = logging.getLogger(__name__)


def get_all_medicines():
    """获取所有药品列表"""
    db = get_db()
    try:
        rows = db.execute(
            "SELECT id, name, category, expire_date, stock, description "
            "FROM medicine ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("药品列表查询失败: %s", e)
        return []


def get_medicine_by_id(medicine_id):
    """根据 ID 获取药品信息"""
    db = get_db()
    try:
        row = db.execute(
            "SELECT id, name, category, expire_date, stock, description "
            "FROM medicine WHERE id = ?",
            (medicine_id,)
        ).fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.warning("药品查询失败 (id=%s): %s", medicine_id, e)
        return None


def add_medicine(name, category=None, expire_date=None, stock=0, description=None):
    """新增药品"""
    db = get_db()
    try:
        cur = db.execute(
            "INSERT INTO medicine (name, category, expire_date, stock, description) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, category, expire_date, stock, description)
        )
        db.commit()
        return cur.lastrowid
    except Exception as e:
        logger.warning("药品添加失败: %s", e)
        return None


def update_medicine(medicine_id, **kwargs):
    """更新药品信息"""
    db = get_db()
    allowed = {"name", "category", "expire_date", "stock", "description"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return False

    try:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [medicine_id]
        cursor = db.execute(f"UPDATE medicine SET {set_clause} WHERE id = ?", values)
        db.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.warning("药品更新失败 (id=%s): %s", medicine_id, e)
        return False


def delete_medicine(medicine_id):
    """删除药品"""
    db = get_db()
    try:
        cursor = db.execute("DELETE FROM medicine WHERE id = ?", (medicine_id,))
        db.commit()
        return cursor.rowcount > 0
    except Exception as e:
        logger.warning("药品删除失败 (id=%s): %s", medicine_id, e)
        return False
