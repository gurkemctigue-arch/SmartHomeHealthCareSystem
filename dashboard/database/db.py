"""数据库初始化与管理"""

import sqlite3
import os
from flask import g


def get_db_path(app) -> str:
    """获取数据库文件路径"""
    return app.config.get("DATABASE", os.path.join(app.instance_path, "medpro.db"))


def get_db(app=None):
    """获取数据库连接（请求上下文内复用）"""
    if app is None:
        from flask import current_app
        app = current_app

    if "db" not in g:
        db_path = get_db_path(app)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
        g.db.execute("PRAGMA foreign_keys=ON")

    return g.db


def close_db(_exception=None):
    """关闭数据库连接"""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    """初始化数据库表结构并注册清理回调"""
    db_path = get_db_path(app)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()

    app.teardown_appcontext(close_db)


# ── 建表 SQL ──────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS medicine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    expire_date TEXT,
    stock INTEGER DEFAULT 0,
    description TEXT
);

CREATE TABLE IF NOT EXISTS detection_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine_name TEXT,
    confidence REAL,
    emotion TEXT,
    emotion_confidence REAL,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS alert_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT,
    title TEXT,
    content TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS chat_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 索引（加速按时间查询）
CREATE INDEX IF NOT EXISTS idx_detection_date ON detection_record(created_at);
CREATE INDEX IF NOT EXISTS idx_alert_date ON alert_record(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_date ON chat_record(created_at);
"""
