import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))

from app import create_app  # noqa: E402


class DashboardApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app({
            "TESTING": True,
            "DATABASE": str(Path(self.temp_dir.name) / "test.db"),
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_index_contains_all_workspaces(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for page in ("overview", "wellness", "detection", "medicine", "emotion", "stats", "alerts", "chat", "settings"):
            self.assertIn(f'data-page-panel="{page}"', html)

    def test_medicine_crud_and_alert_lifecycle(self):
        invalid = self.client.post("/api/medicines", json={"name": "", "stock": 1})
        self.assertEqual(invalid.status_code, 400)

        created = self.client.post("/api/medicines", json={
            "name": "测试药品",
            "category": "测试类别",
            "expire_date": "2020-01-01",
            "stock": 1,
            "description": "接口测试",
        })
        self.assertEqual(created.status_code, 201)
        medicine_id = created.get_json()["data"]["id"]

        medicines = self.client.get("/api/medicines").get_json()["data"]
        self.assertEqual(len(medicines), 1)
        self.assertEqual(medicines[0]["name"], "测试药品")

        open_alerts = self.client.get("/api/alerts?status=open").get_json()["data"]
        self.assertEqual({alert["title"] for alert in open_alerts}, {"药品过期", "库存不足"})

        acknowledged = self.client.post(f"/api/alerts/{open_alerts[0]['id']}/acknowledge")
        self.assertEqual(acknowledged.status_code, 200)

        updated = self.client.put(f"/api/medicines/{medicine_id}", json={
            "name": "测试药品",
            "category": "测试类别",
            "expire_date": "2099-01-01",
            "stock": 10,
            "description": "已补充库存",
        })
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(self.client.get("/api/alerts?status=open").get_json()["data"], [])

        deleted = self.client.delete(f"/api/medicines/{medicine_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/medicines").get_json()["data"], [])

    def test_detection_and_emotion_empty_states(self):
        detections = self.client.get("/api/detections/latest").get_json()["data"]
        emotions = self.client.get("/api/emotions/summary").get_json()["data"]
        self.assertEqual(detections, [])
        self.assertIsNone(emotions["latest"])
        self.assertEqual(emotions["distribution"], [])

    def test_existing_alert_table_is_migrated(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        connection = sqlite3.connect(legacy_path)
        connection.execute(
            "CREATE TABLE alert_record ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, level TEXT, title TEXT, "
            "content TEXT, created_at TEXT)"
        )
        connection.commit()
        connection.close()

        migrated_app = create_app({"TESTING": True, "DATABASE": str(legacy_path)})
        with migrated_app.app_context():
            from database.db import get_db

            columns = {
                row[1] for row in get_db().execute("PRAGMA table_info(alert_record)").fetchall()
            }
        self.assertIn("status", columns)
        self.assertIn("acknowledged_at", columns)


if __name__ == "__main__":
    unittest.main()
