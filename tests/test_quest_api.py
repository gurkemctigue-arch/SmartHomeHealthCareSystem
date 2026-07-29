import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))

from app import create_app  # noqa: E402
from services.quest_content import GAME_CASES  # noqa: E402


class QuestApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app({
            "TESTING": True,
            "DATABASE": str(Path(self.temp_dir.name) / "quest.db"),
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_seeded_overview_and_articles_are_public_safe(self):
        overview = self.client.get("/api/quest/overview")
        self.assertEqual(overview.status_code, 200)
        data = overview.get_json()["data"]
        self.assertEqual(data["profile"]["display_name"], "MedPro 探索者")
        self.assertEqual(data["stats"]["articles_total"], 8)
        self.assertEqual(data["profile"]["xp"], 0)

        articles = self.client.get("/api/quest/articles").get_json()["data"]["items"]
        self.assertEqual(len(articles), 8)
        detail = self.client.get(f"/api/quest/articles/{articles[0]['slug']}").get_json()["data"]
        self.assertEqual(len(detail["quiz"]), 3)
        self.assertNotIn("answer", detail["quiz"][0])
        self.assertTrue(detail["source_url"].startswith("https://"))

    def test_reading_and_quiz_points_cannot_be_farmed(self):
        slug = "antibiotics-are-not-cold-medicine"
        first = self.client.post(f"/api/quest/articles/{slug}/progress", json={"progress": 95})
        second = self.client.post(f"/api/quest/articles/{slug}/progress", json={"progress": 100})
        self.assertEqual(first.get_json()["data"]["points_awarded"], 5)
        self.assertEqual(second.get_json()["data"]["points_awarded"], 0)
        self.assertIn("first_signal", {item["key"] for item in first.get_json()["data"]["unlocked"]})

        quiz_first = self.client.post(f"/api/quest/articles/{slug}/quiz", json={"answers": [0, 1, 2]})
        quiz_second = self.client.post(f"/api/quest/articles/{slug}/quiz", json={"answers": [0, 1, 2]})
        self.assertEqual(quiz_first.get_json()["data"]["score"], 100)
        self.assertEqual(quiz_first.get_json()["data"]["points_awarded"], 15)
        self.assertEqual(quiz_second.get_json()["data"]["points_awarded"], 0)

        overview = self.client.get("/api/quest/overview").get_json()["data"]
        self.assertEqual(overview["profile"]["xp"], 55)

    def test_comments_are_scoped_and_medical_claims_are_reviewed(self):
        slug = "five-checks-before-taking-medicine"
        published = self.client.post(
            f"/api/quest/articles/{slug}/comments",
            json={"content": "这篇文章让我开始按通用名称整理家庭药箱。"},
        )
        reviewed = self.client.post(
            f"/api/quest/articles/{slug}/comments",
            json={"content": "我建议每天 6 片，保证治愈，不用看医生。"},
        )
        self.assertEqual(published.status_code, 201)
        self.assertEqual(published.get_json()["data"]["status"], "published")
        self.assertEqual(published.get_json()["data"]["points_awarded"], 2)
        self.assertEqual(reviewed.get_json()["data"]["status"], "review")
        self.assertEqual(reviewed.get_json()["data"]["points_awarded"], 0)

        detail = self.client.get(f"/api/quest/articles/{slug}").get_json()["data"]
        self.assertEqual(len(detail["comments"]), 2)
        self.assertEqual({item["status"] for item in detail["comments"]}, {"published", "review"})

    def test_profiles_keep_progress_separate(self):
        self.client.post(
            "/api/quest/articles/antibiotics-are-not-cold-medicine/progress",
            json={"progress": 95},
        )
        created = self.client.post("/api/quest/profiles", json={"display_name": "第二位探索者"})
        self.assertEqual(created.status_code, 201)
        second_profile = created.get_json()["data"]
        second_overview = self.client.get("/api/quest/overview").get_json()["data"]
        self.assertEqual(second_overview["profile"]["id"], second_profile["id"])
        self.assertEqual(second_overview["profile"]["xp"], 0)
        self.assertEqual(second_overview["stats"]["articles_completed"], 0)

        activated = self.client.post("/api/quest/profiles/1/activate")
        self.assertEqual(activated.status_code, 200)
        first_overview = self.client.get("/api/quest/overview").get_json()["data"]
        self.assertEqual(first_overview["stats"]["articles_completed"], 1)

    def test_game_is_server_scored_and_awards_safe_shift(self):
        started = self.client.post("/api/quest/game/sessions")
        self.assertEqual(started.status_code, 201)
        session = started.get_json()["data"]
        self.assertEqual(len(session["cases"]), 8)
        self.assertNotIn("correct_action", session["cases"][0])
        solutions = {case["id"]: case["correct_action"] for case in GAME_CASES}

        last_result = None
        for case in session["cases"]:
            response = self.client.post(
                f"/api/quest/game/sessions/{session['session_id']}/action",
                json={"case_id": case["id"], "action": solutions[case["id"]]},
            )
            self.assertEqual(response.status_code, 200)
            last_result = response.get_json()["data"]
            self.assertTrue(last_result["correct"])
            self.assertFalse(last_result["unsafe"])

        duplicate = self.client.post(
            f"/api/quest/game/sessions/{session['session_id']}/action",
            json={"case_id": session["cases"][0]["id"], "action": solutions[session["cases"][0]["id"]]},
        )
        self.assertEqual(duplicate.status_code, 400)

        finished = self.client.post(f"/api/quest/game/sessions/{session['session_id']}/finish")
        self.assertEqual(finished.status_code, 200)
        summary = finished.get_json()["data"]
        self.assertEqual(summary["safety_score"], 100)
        self.assertEqual(summary["correct_actions"], 8)
        self.assertGreater(summary["points_awarded"], 0)
        unlocked = {item["key"] for item in summary["unlocked"]}
        self.assertTrue({"first_shift", "safety_first"} <= unlocked)
        self.assertGreaterEqual(summary["score"], last_result["score"])


if __name__ == "__main__":
    unittest.main()
