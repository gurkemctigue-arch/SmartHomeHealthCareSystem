import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))

from app import create_app  # noqa: E402
from services.weather_service import (  # noqa: E402
    WeatherProvider,
    WeatherServiceError,
    get_weather_snapshot,
)
from services.recipe_service import list_recipe_ids  # noqa: E402


def sample_snapshot(aqi=128, temperature=32):
    return {
        "location": {
            "name": "测试市", "admin1": "测试省", "country": "中国",
            "latitude": 30.1, "longitude": 120.2, "timezone": "Asia/Shanghai",
        },
        "current": {
            "time": "2026-07-29T10:00", "temperature": temperature,
            "feels_like": temperature + 2, "humidity": 76, "weather_code": 1,
            "condition": "少云", "weather_group": "cloud", "wind_speed": 9.2,
            "precipitation": 0, "is_day": True, "aqi": aqi,
            "aqi_label": "轻度污染", "aqi_level": "elevated",
            "pm2_5": 48.0, "pm10": 65.0, "pollen": None,
        },
        "today": {
            "weather_code": 80, "temperature_max": 35, "temperature_min": 27,
            "sunrise": "2026-07-29T05:12", "sunset": "2026-07-29T19:20",
            "uv_index_max": 8.2, "precipitation_probability_max": 72,
        },
        "hourly": [
            {"time": f"2026-07-29T{hour:02d}:00", "temperature": 28 + hour / 10,
             "precipitation_probability": 20 + hour, "uv_index": 2,
             "weather_code": 1}
            for hour in range(24)
        ],
        "provenance": {
            "provider": "Open-Meteo", "geocoder": "OpenStreetMap Nominatim",
            "fetched_at": "2026-07-29T02:00:00+00:00", "state": "live",
            "components": {"weather": "live", "air": "live"},
        },
        "sources": [{"name": "Open-Meteo Weather", "url": "https://open-meteo.com/"}],
    }


class FakeProvider(WeatherProvider):
    name = "fake-weather"

    def __init__(self):
        self.fail = False
        self.forecast_calls = 0
        self.air_calls = 0

    def forecast(self, latitude, longitude):
        self.forecast_calls += 1
        if self.fail:
            raise WeatherServiceError("offline")
        return {
            "timezone": "Asia/Shanghai",
            "current": {
                "time": "2026-07-29T10:00", "temperature_2m": 31,
                "apparent_temperature": 34, "relative_humidity_2m": 62,
                "weather_code": 0, "wind_speed_10m": 8, "precipitation": 0,
                "is_day": 1,
            },
            "hourly": {
                "time": ["2026-07-29T10:00"], "temperature_2m": [31],
                "precipitation_probability": [10], "weather_code": [0], "uv_index": [6],
            },
            "daily": {
                "weather_code": [0], "temperature_2m_max": [35],
                "temperature_2m_min": [27], "sunrise": ["2026-07-29T05:10"],
                "sunset": ["2026-07-29T19:30"], "uv_index_max": [8],
                "precipitation_probability_max": [20],
            },
        }

    def air_quality(self, latitude, longitude):
        self.air_calls += 1
        if self.fail:
            raise WeatherServiceError("offline")
        return {"current": {"us_aqi": 88, "pm2_5": 28, "pm10": 42}}

    def search_locations(self, query, limit=6):
        return []

    def reverse_geocode(self, latitude, longitude):
        if self.fail:
            raise WeatherServiceError("offline")
        return {"name": "缓存测试市", "admin1": "", "country": "中国"}


class WellnessApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app({
            "TESTING": True,
            "DATABASE": str(Path(self.temp_dir.name) / "test.db"),
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_health_profile_roundtrip_and_validation(self):
        default_profile = self.client.get("/api/wellness/profile").get_json()["data"]
        self.assertEqual(default_profile["age_group"], "adult")
        self.assertFalse(default_profile["retain_location"])

        payload = {
            "age_group": "older_adult", "sex": "female",
            "conditions": ["hypertension", "asthma"],
            "allergies": ["seafood", "dairy"],
            "medications": "测试用药",
            "dietary_preferences": ["low_sodium"],
            "health_goals": ["blood_pressure"],
            "retain_location": True,
        }
        updated = self.client.put("/api/wellness/profile", json=payload)
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.get_json()["data"]["conditions"], ["hypertension", "asthma"])
        self.assertTrue(self.client.get("/api/wellness/profile").get_json()["data"]["retain_location"])

        invalid = self.client.put("/api/wellness/profile", json={"conditions": ["unknown"]})
        self.assertEqual(invalid.status_code, 400)

    def test_daily_recommendation_filters_profile_constraints(self):
        self.client.put("/api/wellness/profile", json={
            "age_group": "adult", "sex": "unspecified",
            "conditions": ["asthma", "gout"],
            "allergies": ["seafood", "dairy"],
            "medications": "", "dietary_preferences": ["low_sodium"],
            "health_goals": [], "retain_location": False,
        })
        with patch("app.get_weather_snapshot", return_value=sample_snapshot()):
            response = self.client.get("/api/wellness/daily?latitude=30.1&longitude=120.2")
            second = self.client.get("/api/wellness/daily?latitude=30.1&longitude=120.2")

        self.assertEqual(response.status_code, 200)
        data = response.get_json()["data"]
        recommendation = data["recommendation"]
        self.assertIn("海鲜", recommendation["excluded"])
        self.assertIn("乳制品", recommendation["excluded"])
        self.assertNotIn("鱼", recommendation["meals"][1]["title"])
        self.assertIn("lunch-vegetable-bowl.jpg", recommendation["meals"][1]["image"])
        air_risk = next(item for item in recommendation["risks"] if item["key"] == "air")
        self.assertEqual(air_risk["level"], "high")
        self.assertEqual(second.get_json()["data"]["recommendation"]["meta"]["cache"], "cache")

    def test_city_search_contract(self):
        locations = [{
            "name": "杭州", "admin1": "浙江", "country": "中国",
            "latitude": 30.27, "longitude": 120.15, "timezone": "Asia/Shanghai",
        }]
        with patch("app.search_locations", return_value=locations):
            response = self.client.get("/api/wellness/locations?q=杭州")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["data"][0]["name"], "杭州")
        self.assertEqual(self.client.get("/api/wellness/locations?q=a").status_code, 400)

    def test_weather_cache_falls_back_to_stale_real_data(self):
        provider = FakeProvider()
        with self.app.app_context():
            first = get_weather_snapshot(30.1, 120.2, provider=provider)
            provider.fail = True
            fallback = get_weather_snapshot(30.1, 120.2, force=True, provider=provider)

        self.assertEqual(first["provenance"]["state"], "live")
        self.assertEqual(fallback["provenance"]["state"], "stale")
        self.assertEqual(fallback["current"]["temperature"], 31.0)
        self.assertEqual(provider.forecast_calls, 2)
        self.assertEqual(provider.air_calls, 2)

    def test_recipe_catalog_and_detail_contract(self):
        self.assertEqual(len(list_recipe_ids()), 9)
        response = self.client.get("/api/wellness/recipes/berry-oat-cup")
        self.assertEqual(response.status_code, 200)
        recipe = response.get_json()["data"]
        self.assertEqual(recipe["id"], "berry-oat-cup")
        self.assertGreaterEqual(len(recipe["steps"]), 6)
        self.assertIn("nutrition_note", recipe)
        self.assertIn("profile_match", recipe)
        self.assertEqual(self.client.get("/api/wellness/recipes/not-found").status_code, 404)

    def test_recipe_marks_profile_conflicts_ineligible(self):
        self.client.put("/api/wellness/profile", json={
            "age_group": "adult", "sex": "unspecified",
            "conditions": ["gout"], "allergies": ["seafood"],
            "medications": "", "dietary_preferences": ["vegan"],
            "health_goals": [], "retain_location": False,
        })
        recipe = self.client.get("/api/wellness/recipes/pan-seared-fish-bowl").get_json()["data"]
        self.assertFalse(recipe["profile_match"]["eligible"])
        self.assertEqual(recipe["profile_match"]["status"], "ineligible")
        self.assertGreaterEqual(len(recipe["profile_match"]["blockers"]), 3)

    def test_daily_meals_include_recipe_navigation_and_evidence(self):
        with patch("app.get_weather_snapshot", return_value=sample_snapshot()):
            response = self.client.get("/api/wellness/daily?latitude=30.1&longitude=120.2")
        meals = response.get_json()["data"]["recommendation"]["meals"]
        self.assertEqual(len(meals), 3)
        self.assertTrue(all(meal["recipe_id"] for meal in meals))
        self.assertTrue(all(len(meal["evidence"]) == 3 for meal in meals))
        self.assertTrue(all(0 < meal["fit_score"] <= 100 for meal in meals))


if __name__ == "__main__":
    unittest.main()
