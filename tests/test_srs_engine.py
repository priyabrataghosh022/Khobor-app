"""
Unit tests for the SM-2 Spaced Repetition engine.
"""

import unittest
from datetime import datetime
from srs_engine import (
    sm2_update,
    get_initial_srs_state,
    calculate_next_review_date,
    attach_srs_to_card,
)


class TestSM2Engine(unittest.TestCase):

    def test_sm2_first_review_pass(self):
        # Initial: EF=2.5, interval=1, repetitions=0. Quality=4 (pass)
        ef, interval, reps = sm2_update(2.5, 1, 0, 4)
        self.assertEqual(interval, 1)
        self.assertEqual(reps, 1)
        self.assertAlmostEqual(ef, 2.5, places=2)

    def test_sm2_second_review_pass(self):
        # Second pass: reps=1 -> interval becomes 6
        ef, interval, reps = sm2_update(2.5, 1, 1, 4)
        self.assertEqual(interval, 6)
        self.assertEqual(reps, 2)

    def test_sm2_third_review_pass(self):
        # Third pass: reps=2 -> interval = round(6 * 2.5) = 15
        ef, interval, reps = sm2_update(2.5, 6, 2, 4)
        self.assertEqual(interval, 15)
        self.assertEqual(reps, 3)

    def test_sm2_fail_resets_repetitions_and_interval(self):
        # If quality < 3 (fail), reps reset to 0, interval resets to 1
        ef, interval, reps = sm2_update(2.5, 15, 3, 2)
        self.assertEqual(reps, 0)
        self.assertEqual(interval, 1)
        # Ease factor decreases on failure
        self.assertLess(ef, 2.5)

    def test_sm2_minimum_ease_factor_clamp(self):
        # Repeated failures should never let ease factor drop below 1.3
        ef = 1.35
        for _ in range(5):
            ef, _, _ = sm2_update(ef, 1, 0, 0)
        self.assertEqual(ef, 1.3)

    def test_initial_state_generation(self):
        state = get_initial_srs_state("2026-10-15")
        self.assertEqual(state["ease_factor"], 2.5)
        self.assertEqual(state["interval_days"], 1)
        self.assertEqual(state["repetitions"], 0)
        self.assertEqual(state["next_review_date"], "2026-10-15")
        self.assertEqual(state["last_reviewed"], "")

    def test_calculate_next_review_date(self):
        next_date = calculate_next_review_date("2026-10-10", 5)
        self.assertEqual(next_date, "2026-10-15")

    def test_attach_srs_to_card(self):
        card = {"word": "laconic", "meaning": "concise"}
        enriched = attach_srs_to_card(card, "2026-10-11")
        self.assertEqual(enriched["word"], "laconic")
        self.assertEqual(enriched["ease_factor"], 2.5)
        self.assertEqual(enriched["next_review_date"], "2026-10-11")


if __name__ == "__main__":
    unittest.main()
