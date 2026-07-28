-- 多模态智能医疗家庭助手 数据库建表脚本
-- 此文件供手动初始化数据库使用，程序启动时会自动建表

-- 药品信息表
CREATE TABLE IF NOT EXISTS medicine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    category TEXT,
    expire_date TEXT,
    stock INTEGER DEFAULT 0,
    description TEXT
);

-- 检测记录表
CREATE TABLE IF NOT EXISTS detection_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine_name TEXT,
    confidence REAL,
    emotion TEXT,
    emotion_confidence REAL,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 告警记录表
CREATE TABLE IF NOT EXISTS alert_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT,
    title TEXT,
    content TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 问答记录表
CREATE TABLE IF NOT EXISTS chat_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_detection_date ON detection_record(created_at);
CREATE INDEX IF NOT EXISTS idx_alert_date ON alert_record(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_date ON chat_record(created_at);
