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
    status TEXT DEFAULT 'open',
    acknowledged_at TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 问答记录表
CREATE TABLE IF NOT EXISTS chat_record (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 天气与空气质量响应缓存（不存储浏览器定位精度）
CREATE TABLE IF NOT EXISTS weather_cache (
    cache_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    payload TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

-- 本机单用户健康档案
CREATE TABLE IF NOT EXISTS health_profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    age_group TEXT NOT NULL DEFAULT 'adult',
    sex TEXT NOT NULL DEFAULT 'unspecified',
    conditions TEXT NOT NULL DEFAULT '[]',
    allergies TEXT NOT NULL DEFAULT '[]',
    medications TEXT NOT NULL DEFAULT '',
    dietary_preferences TEXT NOT NULL DEFAULT '[]',
    health_goals TEXT NOT NULL DEFAULT '[]',
    retain_location INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
);

-- 每日推荐缓存；天气风险级别或健康档案变化后会生成新键
CREATE TABLE IF NOT EXISTS wellness_recommendation_cache (
    cache_key TEXT PRIMARY KEY,
    recommendation_date TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 家庭生命镜像：成员、体征与照护任务
CREATE TABLE IF NOT EXISTS family_member (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'other',
    sex TEXT NOT NULL DEFAULT 'unspecified',
    birth_date TEXT,
    height_cm REAL,
    color TEXT NOT NULL DEFAULT 'mint',
    conditions TEXT NOT NULL DEFAULT '[]',
    allergies TEXT NOT NULL DEFAULT '[]',
    is_primary INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS health_measurement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id INTEGER NOT NULL,
    metric_type TEXT NOT NULL,
    value_primary REAL NOT NULL,
    value_secondary REAL,
    unit TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    note TEXT,
    measured_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (member_id) REFERENCES family_member(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS health_task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id INTEGER NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'care',
    title TEXT NOT NULL,
    detail TEXT,
    due_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    priority TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'open',
    completed_at TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (member_id) REFERENCES family_member(id) ON DELETE CASCADE
);

-- VITAL QUEST 多用户身份与内容系统
CREATE TABLE IF NOT EXISTS quest_user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    avatar_seed TEXT NOT NULL DEFAULT 'aurora',
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS quest_article (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    dek TEXT NOT NULL,
    category TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    reading_minutes INTEGER NOT NULL DEFAULT 5,
    cover_asset TEXT,
    accent TEXT NOT NULL DEFAULT 'cyan',
    body TEXT NOT NULL,
    takeaways TEXT NOT NULL,
    quiz TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS quest_article_progress (
    user_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    best_quiz_score INTEGER NOT NULL DEFAULT 0,
    completed_at TEXT,
    last_read_at TEXT DEFAULT (datetime('now', 'localtime')),
    PRIMARY KEY (user_id, article_id),
    FOREIGN KEY (user_id) REFERENCES quest_user(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES quest_article(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quest_comment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'published',
    helpful_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES quest_user(id) ON DELETE CASCADE,
    FOREIGN KEY (article_id) REFERENCES quest_article(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quest_points_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    event_key TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    reference_type TEXT,
    reference_id TEXT,
    points INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (user_id) REFERENCES quest_user(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quest_achievement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    achievement_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    icon TEXT NOT NULL,
    tone TEXT NOT NULL,
    bonus_points INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS quest_user_achievement (
    user_id INTEGER NOT NULL,
    achievement_id INTEGER NOT NULL,
    unlocked_at TEXT DEFAULT (datetime('now', 'localtime')),
    PRIMARY KEY (user_id, achievement_id),
    FOREIGN KEY (user_id) REFERENCES quest_user(id) ON DELETE CASCADE,
    FOREIGN KEY (achievement_id) REFERENCES quest_achievement(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS quest_game_session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    public_id TEXT NOT NULL UNIQUE,
    user_id INTEGER NOT NULL,
    scenario_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    event_log TEXT NOT NULL DEFAULT '[]',
    score INTEGER NOT NULL DEFAULT 0,
    safety_score INTEGER NOT NULL DEFAULT 100,
    patients_served INTEGER NOT NULL DEFAULT 0,
    correct_actions INTEGER NOT NULL DEFAULT 0,
    unsafe_actions INTEGER NOT NULL DEFAULT 0,
    duration_seconds INTEGER NOT NULL DEFAULT 0,
    started_at TEXT DEFAULT (datetime('now', 'localtime')),
    completed_at TEXT,
    FOREIGN KEY (user_id) REFERENCES quest_user(id) ON DELETE CASCADE
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_detection_date ON detection_record(created_at);
CREATE INDEX IF NOT EXISTS idx_alert_date ON alert_record(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_date ON chat_record(created_at);
CREATE INDEX IF NOT EXISTS idx_wellness_recommendation_date
    ON wellness_recommendation_cache(recommendation_date);
CREATE INDEX IF NOT EXISTS idx_health_measurement_member_date
    ON health_measurement(member_id, measured_at);
CREATE INDEX IF NOT EXISTS idx_health_task_member_status
    ON health_task(member_id, status, due_at);
CREATE INDEX IF NOT EXISTS idx_quest_article_category ON quest_article(category, status);
CREATE INDEX IF NOT EXISTS idx_quest_comment_article ON quest_comment(article_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quest_ledger_user_date ON quest_points_ledger(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_quest_game_user_date ON quest_game_session(user_id, started_at);
