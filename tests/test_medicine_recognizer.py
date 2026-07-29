from __future__ import annotations

import unittest
from concurrent.futures import Future
from types import SimpleNamespace

import cv2
import numpy as np

from dashboard.models.medicine_recognizer import (
    Detection,
    MedicineRecognizer,
    TrackState,
    _appearance_distance,
    _appearance_signature,
)


def _state(status: str = "pending") -> TrackState:
    return TrackState(
        state_id=1,
        bbox=(0.0, 0.0, 100.0, 60.0),
        last_seen=0,
        status=status,
    )


def _match(medicine_id: int, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        medicine_id=medicine_id,
        medicine_name=name,
        category_name="测试类别",
        efficacy=f"{name}的测试功效",
        efficacy_source="test",
        score=0.94,
        match_type="fuzzy",
    )


def _recognized(match: SimpleNamespace | None, signature=None) -> dict:
    return {
        "text": match.medicine_name if match else "模糊文字",
        "average_confidence": 0.86,
        "lines": [{"text": "模糊文字", "confidence": 0.86}],
        "_medicine_match": match,
        "_appearance_signature": signature,
    }


class _ImmediateExecutor:
    def __init__(self) -> None:
        self.tasks = []

    def submit(self, function, tasks):
        self.tasks = list(tasks)
        future = Future()
        future.set_result([])
        return future


class MedicineRecognizerSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recognizer = MedicineRecognizer.__new__(MedicineRecognizer)
        self.recognizer.switch_confirm_hits = 2
        self.recognizer.other_category = "其他"
        self.recognizer.other_efficacy = "其他功效"

    def test_first_catalog_match_locks_result(self) -> None:
        state = _state()
        signature = np.ones((4, 108), dtype=np.float32) / 108

        self.recognizer._apply_ocr_result(
            state, _recognized(_match(10, "复方板蓝根颗粒"), signature)
        )

        self.assertEqual(state.status, "known")
        self.assertEqual(state.medicine_id, 10)
        self.assertEqual(state.medicine_name, "复方板蓝根颗粒")
        np.testing.assert_array_equal(state.appearance_signature, signature)

    def test_periodic_different_match_requires_two_results(self) -> None:
        state = _state("known")
        state.medicine_id = 10
        state.medicine_name = "复方板蓝根颗粒"
        new_match = _match(20, "藿香正气水")

        self.recognizer._apply_ocr_result(state, _recognized(new_match))
        self.assertEqual(state.medicine_id, 10)
        self.assertEqual(state.candidate_match_hits, 1)

        self.recognizer._apply_ocr_result(state, _recognized(new_match))
        self.assertEqual(state.medicine_id, 20)
        self.assertEqual(state.medicine_name, "藿香正气水")
        self.assertEqual(state.candidate_match_hits, 0)

    def test_confirmed_appearance_change_switches_on_first_match(self) -> None:
        state = _state("known")
        state.medicine_id = 10
        state.medicine_name = "复方板蓝根颗粒"
        state.appearance_changed = True

        self.recognizer._apply_ocr_result(
            state, _recognized(_match(20, "藿香正气水"))
        )

        self.assertEqual(state.medicine_id, 20)
        self.assertFalse(state.appearance_changed)

    def test_uncertain_ocr_does_not_replace_locked_result(self) -> None:
        state = _state("known")
        state.medicine_id = 10
        state.medicine_name = "复方板蓝根颗粒"

        self.recognizer._apply_ocr_result(state, _recognized(None))

        self.assertEqual(state.status, "known")
        self.assertEqual(state.medicine_name, "复方板蓝根颗粒")

    def test_changed_package_triggers_ocr_without_waiting_for_interval(self) -> None:
        original = np.zeros((80, 120, 3), dtype=np.uint8)
        original[:] = (0, 0, 220)
        replacement = np.zeros_like(original)
        replacement[:] = (220, 0, 0)
        cv2.rectangle(replacement, (0, 0), (55, 79), (0, 220, 0), -1)

        state = _state("known")
        state.medicine_id = 10
        state.appearance_signature = _appearance_signature(original)
        state.appearance_change_hits = 1
        state.last_appearance_check = 95
        state.last_submitted = 99
        state.last_completed = 99

        executor = _ImmediateExecutor()
        self.recognizer._future = None
        self.recognizer._executor = executor
        self.recognizer._states = {1: state}
        self.recognizer._frame_index = 100
        self.recognizer.switch_check_interval = 4
        self.recognizer.switch_threshold = 0.28
        self.recognizer.ocr_interval = 8
        self.recognizer.ocr_known_interval = 90
        self.recognizer.ocr_max_boxes = 1
        self.recognizer.min_ocr_area = 100
        self.recognizer._perspective_crop = lambda *_: replacement

        polygon = np.asarray(
            [[0, 0], [120, 0], [120, 80], [0, 80]], dtype=np.float32
        )
        detection = Detection(
            polygon_norm=polygon / np.asarray([120, 80], dtype=np.float32),
            polygon=polygon,
            confidence=0.8,
            state_id=1,
        )

        self.recognizer._submit_ocr(replacement, [detection])

        self.assertTrue(state.appearance_changed)
        self.assertEqual(len(executor.tasks), 1)

    def test_color_signature_separates_different_packages(self) -> None:
        red = np.full((80, 120, 3), (0, 0, 220), dtype=np.uint8)
        brighter_red = np.full((80, 120, 3), (0, 0, 255), dtype=np.uint8)
        blue_green = np.full((80, 120, 3), (220, 0, 0), dtype=np.uint8)
        cv2.rectangle(blue_green, (0, 0), (55, 79), (0, 220, 0), -1)

        same_distance = _appearance_distance(
            _appearance_signature(red), _appearance_signature(brighter_red)
        )
        different_distance = _appearance_distance(
            _appearance_signature(red), _appearance_signature(blue_green)
        )

        self.assertLess(same_distance, 0.28)
        self.assertGreater(different_distance, 0.28)


if __name__ == "__main__":
    unittest.main()
