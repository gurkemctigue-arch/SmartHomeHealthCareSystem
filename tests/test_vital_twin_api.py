import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))

from app import create_app  # noqa: E402


class VitalTwinApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app({
            "TESTING": True,
            "DATABASE": str(Path(self.temp_dir.name) / "vital-twin.db"),
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_member_and_empty_baseline(self):
        response = self.client.get("/api/vital-twin/overview")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()["data"]
        self.assertEqual(data["member"]["name"], "我")
        self.assertEqual(data["score"], None)
        self.assertEqual(data["coverage"], 0)
        self.assertEqual(data["timeline"], [])
        self.assertTrue(any(item["kind"] == "baseline" for item in data["actions"]))
        self.assertTrue(all(item["status"] == "baseline" for item in data["signals"]))

    def test_member_crud_and_last_member_guard(self):
        created = self.client.post("/api/vital-twin/members", json={
            "name": "妈妈",
            "relation": "mother",
            "sex": "female",
            "birth_date": "1968-04-16",
            "height_cm": 162,
            "conditions": ["高血压"],
            "allergies": ["青霉素"],
            "color": "coral",
        })
        self.assertEqual(created.status_code, 201)
        member = created.get_json()["data"]
        self.assertEqual(member["relation_label"], "母亲")
        self.assertEqual(member["conditions"], ["高血压"])

        updated = self.client.put(f"/api/vital-twin/members/{member['id']}", json={
            "name": "母亲",
            "relation": "mother",
            "sex": "female",
            "birth_date": "1968-04-16",
            "height_cm": 162,
            "conditions": ["高血压", "高血脂"],
            "allergies": [],
            "color": "amber",
        })
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.get_json()["data"]["conditions"], ["高血压", "高血脂"])

        self.assertEqual(self.client.delete(f"/api/vital-twin/members/{member['id']}").status_code, 200)
        only_member = self.client.get("/api/vital-twin/members").get_json()["data"][0]
        guarded = self.client.delete(f"/api/vital-twin/members/{only_member['id']}")
        self.assertEqual(guarded.status_code, 400)

    def test_measurement_updates_body_signal_actions_and_timeline(self):
        member_id = self.client.get("/api/vital-twin/members").get_json()["data"][0]["id"]
        measurement = self.client.post("/api/vital-twin/measurements", json={
            "member_id": member_id,
            "metric_type": "blood_pressure",
            "value_primary": 188,
            "value_secondary": 122,
            "measured_at": datetime.now().isoformat(timespec="minutes"),
            "note": "复测记录",
        })
        self.assertEqual(measurement.status_code, 201)
        self.assertEqual(measurement.get_json()["data"]["status"], "danger")

        data = self.client.get(f"/api/vital-twin/overview?member_id={member_id}").get_json()["data"]
        cardio = next(item for item in data["signals"] if item["key"] == "cardio")
        self.assertEqual(cardio["status"], "danger")
        self.assertEqual(data["score"], 80)
        self.assertEqual(data["timeline"][0]["metric_type"], "blood_pressure")
        self.assertTrue(any(item["priority"] == "danger" for item in data["actions"]))

    def test_measurement_and_member_validation(self):
        member_id = self.client.get("/api/vital-twin/members").get_json()["data"][0]["id"]
        invalid_pressure = self.client.post("/api/vital-twin/measurements", json={
            "member_id": member_id,
            "metric_type": "blood_pressure",
            "value_primary": 120,
            "value_secondary": 130,
        })
        self.assertEqual(invalid_pressure.status_code, 400)
        self.assertEqual(self.client.get("/api/vital-twin/overview?member_id=abc").status_code, 400)
        self.assertEqual(self.client.get("/api/vital-twin/overview?member_id=9999").status_code, 404)

    def test_care_task_lifecycle(self):
        member_id = self.client.get("/api/vital-twin/members").get_json()["data"][0]["id"]
        created = self.client.post("/api/vital-twin/tasks", json={
            "member_id": member_id,
            "title": "晚间测量血压",
            "detail": "静坐五分钟后测量",
            "priority": "important",
            "due_at": (datetime.now() + timedelta(hours=2)).isoformat(timespec="minutes"),
        })
        self.assertEqual(created.status_code, 201)
        task_id = created.get_json()["data"]["id"]
        actions = self.client.get("/api/vital-twin/overview").get_json()["data"]["actions"]
        self.assertTrue(any(item.get("task_id") == task_id for item in actions))

        completed = self.client.post(f"/api/vital-twin/tasks/{task_id}/complete")
        self.assertEqual(completed.status_code, 200)
        data = self.client.get("/api/vital-twin/overview").get_json()["data"]
        self.assertFalse(any(item.get("task_id") == task_id for item in data["actions"]))
        task_event = next(item for item in data["timeline"] if item["id"] == f"task-{task_id}")
        self.assertEqual(task_event["status"], "good")

    def test_index_includes_vital_twin_workspace(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('data-page="vital"', html)
        self.assertIn('data-page-panel="vital"', html)
        self.assertIn('id="vitalTwinCanvas"', html)
        self.assertIn("js/vital-twin.js", html)


if __name__ == "__main__":
    unittest.main()
