import sys
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AI_Training"))

from analyze_card_ab import compare_arms


class AnalyzeCardABTests(unittest.TestCase):
    @patch("analyze_card_ab.card_shadow_rows")
    @patch("analyze_card_ab.latest_runs")
    def test_compare_arms_summarizes_outcomes_and_card_metrics(self, latest_runs_mock, card_rows_mock):
        latest_runs_mock.return_value = [
            {"run_id": "b1", "seed": "101", "max_floor": 12, "max_act": 1, "boss_damage": 0, "invalid_actions": 0},
            {"run_id": "b2", "seed": "202", "max_floor": 18, "max_act": 2, "boss_damage": 10, "invalid_actions": 0},
            {"run_id": "c1", "seed": "101", "max_floor": 20, "max_act": 2, "boss_damage": 20, "invalid_actions": 0},
            {"run_id": "c2", "seed": "202", "max_floor": 22, "max_act": 2, "boss_damage": 30, "invalid_actions": 0},
        ]
        card_rows_mock.return_value = [
            {"type": "card_scorer_shadow", "run_id": "b1", "recommended_action": "choose_card:index_0", "legacy_chosen_action": "choose_card:index_0", "deck_summary": {"deck_size": 14}},
            {"type": "card_scorer_shadow", "run_id": "b2", "recommended_action": "choose_card:index_1", "legacy_chosen_action": "choose_card:index_1", "deck_summary": {"deck_size": 16}},
            {"type": "card_scorer_shadow", "run_id": "c1", "seed": "101", "floor": 9, "screen_type": "card_reward", "template_id": "barricade_block", "recommended_action": "skip_reward", "legacy_chosen_action": "choose_card:index_0", "old_policy_action": "choose_card:index_0", "final_executed_action": "skip_reward", "old_policy_card": {"name": "Old"}, "scorer_card": {"name": "Skip"}, "confidence_gap": 1.2, "deck_summary": {"deck_size": 18}, "canary_takeover": True, "executed_skip": True},
            {"type": "card_scorer_shadow", "run_id": "c2", "recommended_action": "choose_card:index_2", "legacy_chosen_action": "choose_card:index_2", "old_policy_action": "choose_card:index_2", "final_executed_action": "choose_card:index_2", "deck_summary": {"deck_size": 20}, "canary_takeover": False},
        ]

        summary = compare_arms(["b1", "b2"], ["c1", "c2"])

        self.assertEqual(summary["baseline"]["run_count"], 2)
        self.assertEqual(summary["active_canary"]["run_count"], 2)
        self.assertEqual(summary["active_canary"]["takeover_count"], 1)
        self.assertEqual(summary["active_canary"]["scorer_skip_rate"], 0.5)
        self.assertEqual(summary["active_canary"]["executed_skip_rate"], 0.5)
        self.assertEqual(summary["delta"]["average_floor"], 6.0)
        self.assertEqual(summary["delta"]["act1_clear_rate"], 0.5)
        self.assertEqual(summary["per_seed"][0]["seed"], "101")
        self.assertEqual(summary["per_seed"][0]["floor_delta"], 8)
        self.assertEqual(summary["per_seed"][0]["takeover_count"], 1)
        self.assertEqual(summary["per_seed"][0]["skip_takeover_count"], 1)
        self.assertEqual(summary["per_seed"][0]["pick_takeover_count"], 0)
        self.assertEqual(summary["final_action_first_divergence_count"], 1)
        self.assertEqual(summary["takeover_examples"][0]["old_card_name"], "Old")


if __name__ == "__main__":
    unittest.main()
