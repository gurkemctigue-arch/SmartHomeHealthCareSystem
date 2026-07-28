"""告警服务"""

from datetime import datetime


def get_latest_alerts(limit=5):
    """获取最新告警列表

    后续可从 alert_record 表查询。

    Args:
        limit: 返回条数上限

    Returns:
        list[dict]: 告警记录列表
    """
    now = datetime.now()

    # 模拟告警数据（后续从数据库读取）
    all_alerts = [
        {
            "level": "danger",
            "title": "药品过期警告",
            "content": "布洛芬片已过期，请勿继续服用",
            "time": _format_time(now, -120)
        },
        {
            "level": "warning",
            "title": "药量库存不足",
            "content": "阿莫西林胶囊剩余2盒",
            "time": _format_time(now, -600)
        },
        {
            "level": "warning",
            "title": "情绪异常",
            "content": "检测到低落情绪，建议关注",
            "time": _format_time(now, -1800)
        },
        {
            "level": "info",
            "title": "检测到未知药品",
            "content": "请确认后录入系统",
            "time": _format_time(now, -3600)
        },
        {
            "level": "success",
            "title": "数据入库成功",
            "content": "检测记录已保存",
            "time": _format_time(now, -7200)
        },
    ]

    return all_alerts[:limit]


def _format_time(base_time, offset_seconds):
    """格式化时间"""
    t = base_time if offset_seconds == 0 else datetime.fromtimestamp(
        base_time.timestamp() + offset_seconds
    )
    return t.strftime("%H:%M:%S")


def check_alerts(medicine=None, emotion=None):
    """根据药品和情绪状态生成告警规则

    Args:
        medicine: 药品信息字典
        emotion: 情绪结果字典

    Returns:
        list[dict]: 触发的告警列表
    """
    alerts = []

    if medicine:
        if medicine.get("expire_date", "2099-01-01") < datetime.now().strftime("%Y-%m-%d"):
            alerts.append({
                "level": "danger",
                "title": "药品过期警告",
                "content": f"{medicine.get('name', '未知药品')}已过期，请勿服用"
            })

        if medicine.get("stock", 99) <= 2:
            alerts.append({
                "level": "warning",
                "title": "药量库存不足",
                "content": f"{medicine.get('name', '未知药品')}库存不足"
            })

    if emotion:
        if emotion.get("emotion") in ["悲伤", "焦虑", "愤怒"]:
            alerts.append({
                "level": "warning",
                "title": "情绪异常提醒",
                "content": "检测到异常情绪状态，建议家属关注"
            })

    return alerts
