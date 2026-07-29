"""No-key weather, air-quality and geocoding providers with SQLite caching."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import json
import logging

import requests

from database.db import get_db


logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
REVERSE_GEOCODING_URL = "https://nominatim.openstreetmap.org/reverse"
USER_AGENT = "MedPro-Climate-Wellness/1.0 (local educational application)"


class WeatherServiceError(RuntimeError):
    """Raised when real environmental data and cached fallback are unavailable."""


class WeatherProvider(ABC):
    """Provider contract keeps business logic independent from the weather vendor."""

    name = "unknown"

    @abstractmethod
    def forecast(self, latitude, longitude):
        raise NotImplementedError

    @abstractmethod
    def air_quality(self, latitude, longitude):
        raise NotImplementedError

    @abstractmethod
    def search_locations(self, query, limit=6):
        raise NotImplementedError

    @abstractmethod
    def reverse_geocode(self, latitude, longitude):
        raise NotImplementedError


class OpenMeteoProvider(WeatherProvider):
    """Open-Meteo weather/AQ provider with Nominatim reverse geocoding."""

    name = "open-meteo"

    def __init__(self, session=None, timeout=8):
        self.session = session or requests.Session()
        self.timeout = timeout

    def _get_json(self, url, params):
        try:
            response = self.session.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise WeatherServiceError("环境数据服务暂时不可用") from exc
        if not isinstance(payload, dict) or payload.get("error"):
            raise WeatherServiceError(str(payload.get("reason") or "环境数据响应无效"))
        return payload

    def forecast(self, latitude, longitude):
        return self._get_json(FORECAST_URL, {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join((
                "temperature_2m", "apparent_temperature", "relative_humidity_2m",
                "weather_code", "wind_speed_10m", "precipitation", "is_day",
            )),
            "hourly": "temperature_2m,precipitation_probability,weather_code,uv_index",
            "daily": ",".join((
                "weather_code", "temperature_2m_max", "temperature_2m_min",
                "sunrise", "sunset", "uv_index_max", "precipitation_probability_max",
            )),
            "timezone": "auto",
            "forecast_days": 2,
        })

    def air_quality(self, latitude, longitude):
        return self._get_json(AIR_QUALITY_URL, {
            "latitude": latitude,
            "longitude": longitude,
            "current": "us_aqi,pm2_5,pm10,alder_pollen,birch_pollen,grass_pollen,ragweed_pollen",
            "hourly": "us_aqi,pm2_5",
            "timezone": "auto",
            "forecast_days": 2,
        })

    def search_locations(self, query, limit=6):
        payload = self._get_json(GEOCODING_URL, {
            "name": query,
            "count": limit,
            "language": "zh",
            "format": "json",
        })
        locations = []
        for item in payload.get("results") or []:
            locations.append({
                "name": item.get("name") or "未知地点",
                "admin1": item.get("admin1") or "",
                "country": item.get("country") or "",
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "timezone": item.get("timezone") or "",
            })
        return locations

    def reverse_geocode(self, latitude, longitude):
        payload = self._get_json(REVERSE_GEOCODING_URL, {
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "zoom": 10,
            "addressdetails": 1,
            "accept-language": "zh-CN,zh",
        })
        address = payload.get("address") or {}
        name = (
            address.get("city") or address.get("town") or address.get("county")
            or address.get("state") or payload.get("name") or "当前位置"
        )
        return {
            "name": name,
            "admin1": address.get("state") or "",
            "country": address.get("country") or "",
        }


def validate_coordinates(latitude, longitude):
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("经纬度必须是有效数字") from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("经纬度超出有效范围")
    return round(latitude, 5), round(longitude, 5)


def search_locations(query, provider=None):
    query = str(query or "").strip()
    if len(query) < 2:
        raise ValueError("请输入至少 2 个字符的城市名称")
    if len(query) > 60:
        raise ValueError("城市名称不能超过 60 个字符")
    return (provider or OpenMeteoProvider()).search_locations(query)


def get_weather_snapshot(latitude, longitude, preferred_name="", force=False, provider=None):
    latitude, longitude = validate_coordinates(latitude, longitude)
    provider = provider or OpenMeteoProvider()
    location_key = f"{latitude:.2f},{longitude:.2f}"

    forecast, forecast_state, forecast_at = _cached_fetch(
        f"forecast:{provider.name}:{location_key}", provider.name, 15,
        lambda: provider.forecast(latitude, longitude), force,
    )
    air, air_state, air_at = _cached_fetch(
        f"air:{provider.name}:{location_key}", provider.name, 30,
        lambda: provider.air_quality(latitude, longitude), force,
    )

    preferred_name = str(preferred_name or "").strip()[:80]
    if preferred_name:
        location = {"name": preferred_name, "admin1": "", "country": ""}
    else:
        location, _, _ = _cached_fetch(
            f"place:nominatim:{location_key}", "nominatim", 7 * 24 * 60,
            lambda: provider.reverse_geocode(latitude, longitude), False,
            allow_failure=True,
        )
        location = location or {"name": "当前位置", "admin1": "", "country": ""}

    return _normalize_snapshot(
        forecast,
        air,
        location,
        latitude,
        longitude,
        {"weather": forecast_state, "air": air_state},
        max(forecast_at, air_at),
    )


def _cached_fetch(cache_key, provider, ttl_minutes, fetcher, force, allow_failure=False):
    db = get_db()
    row = db.execute(
        "SELECT payload, fetched_at, expires_at FROM weather_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    now = datetime.now(timezone.utc)
    if row and not force:
        try:
            expires_at = datetime.fromisoformat(row["expires_at"])
            if expires_at > now:
                return json.loads(row["payload"]), "cache", row["fetched_at"]
        except (TypeError, ValueError, json.JSONDecodeError):
            row = None

    try:
        payload = fetcher()
        fetched_at = now.isoformat()
        expires_at = (now + timedelta(minutes=ttl_minutes)).isoformat()
        db.execute(
            "INSERT INTO weather_cache (cache_key, provider, payload, fetched_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET "
            "provider=excluded.provider, payload=excluded.payload, "
            "fetched_at=excluded.fetched_at, expires_at=excluded.expires_at",
            (cache_key, provider, json.dumps(payload, ensure_ascii=False), fetched_at, expires_at),
        )
        db.commit()
        return payload, "live", fetched_at
    except WeatherServiceError:
        if row:
            logger.warning("Using stale environmental cache for %s", cache_key)
            return json.loads(row["payload"]), "stale", row["fetched_at"]
        if allow_failure:
            return None, "unavailable", now.isoformat()
        raise


def _normalize_snapshot(forecast, air, location, latitude, longitude, states, fetched_at):
    current = forecast.get("current") or {}
    daily = forecast.get("daily") or {}
    air_current = air.get("current") or {}
    aqi = _number(air_current.get("us_aqi"))
    condition = weather_condition(current.get("weather_code"))

    hourly_raw = forecast.get("hourly") or {}
    times = hourly_raw.get("time") or []
    current_time = str(current.get("time") or "")[:13]
    start_index = next((i for i, value in enumerate(times) if str(value)[:13] >= current_time), 0)
    hourly = []
    for index in range(start_index, min(start_index + 24, len(times))):
        hourly.append({
            "time": times[index],
            "temperature": _list_number(hourly_raw, "temperature_2m", index),
            "precipitation_probability": _list_number(hourly_raw, "precipitation_probability", index),
            "uv_index": _list_number(hourly_raw, "uv_index", index),
            "weather_code": _list_value(hourly_raw, "weather_code", index),
        })

    return {
        "location": {
            **location,
            "latitude": latitude,
            "longitude": longitude,
            "timezone": forecast.get("timezone") or "auto",
        },
        "current": {
            "time": current.get("time"),
            "temperature": _number(current.get("temperature_2m")),
            "feels_like": _number(current.get("apparent_temperature")),
            "humidity": _number(current.get("relative_humidity_2m")),
            "weather_code": current.get("weather_code"),
            "condition": condition,
            "weather_group": weather_group(current.get("weather_code")),
            "wind_speed": _number(current.get("wind_speed_10m")),
            "precipitation": _number(current.get("precipitation")),
            "is_day": bool(current.get("is_day", 1)),
            "aqi": aqi,
            "aqi_label": aqi_label(aqi),
            "aqi_level": aqi_level(aqi),
            "pm2_5": _number(air_current.get("pm2_5")),
            "pm10": _number(air_current.get("pm10")),
            "pollen": max(filter(lambda value: value is not None, (
                _number(air_current.get("alder_pollen")),
                _number(air_current.get("birch_pollen")),
                _number(air_current.get("grass_pollen")),
                _number(air_current.get("ragweed_pollen")),
            )), default=None),
        },
        "today": {
            "weather_code": _list_value(daily, "weather_code", 0),
            "temperature_max": _list_number(daily, "temperature_2m_max", 0),
            "temperature_min": _list_number(daily, "temperature_2m_min", 0),
            "sunrise": _list_value(daily, "sunrise", 0),
            "sunset": _list_value(daily, "sunset", 0),
            "uv_index_max": _list_number(daily, "uv_index_max", 0),
            "precipitation_probability_max": _list_number(daily, "precipitation_probability_max", 0),
        },
        "hourly": hourly,
        "provenance": {
            "provider": "Open-Meteo",
            "geocoder": "OpenStreetMap Nominatim",
            "fetched_at": fetched_at,
            "state": "stale" if "stale" in states.values() else (
                "live" if "live" in states.values() else "cache"
            ),
            "components": states,
        },
        "sources": [
            {"name": "Open-Meteo Weather", "url": "https://open-meteo.com/"},
            {"name": "Open-Meteo Air Quality", "url": "https://open-meteo.com/en/docs/air-quality-api"},
            {"name": "OpenStreetMap contributors", "url": "https://www.openstreetmap.org/copyright"},
        ],
    }


def weather_condition(code):
    code = int(code or 0)
    if code == 0:
        return "晴朗"
    if code in (1, 2):
        return "少云"
    if code == 3:
        return "阴天"
    if code in (45, 48):
        return "有雾"
    if code in (51, 53, 55, 56, 57):
        return "细雨"
    if code in (61, 63, 65, 66, 67, 80, 81, 82):
        return "降雨"
    if code in (71, 73, 75, 77, 85, 86):
        return "降雪"
    if code in (95, 96, 99):
        return "雷暴"
    return "多云"


def weather_group(code):
    code = int(code or 0)
    if code == 0:
        return "clear"
    if code in (1, 2, 3, 45, 48):
        return "cloud"
    if code in (71, 73, 75, 77, 85, 86):
        return "snow"
    if code in (95, 96, 99):
        return "storm"
    return "rain"


def aqi_label(aqi):
    if aqi is None:
        return "暂无"
    if aqi <= 50:
        return "优"
    if aqi <= 100:
        return "良"
    if aqi <= 150:
        return "轻度污染"
    if aqi <= 200:
        return "中度污染"
    if aqi <= 300:
        return "重度污染"
    return "严重污染"


def aqi_level(aqi):
    if aqi is None or aqi <= 50:
        return "low"
    if aqi <= 100:
        return "guarded"
    if aqi <= 150:
        return "elevated"
    return "high"


def _number(value):
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _list_value(payload, key, index):
    values = payload.get(key) or []
    return values[index] if index < len(values) else None


def _list_number(payload, key, index):
    return _number(_list_value(payload, key, index))
