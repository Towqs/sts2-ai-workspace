import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AI_Training"))

from analyze_noop_ab import attach_trace_divergences, first_trace_divergence


class AnalyzeNoopABTests(unittest.TestCase):
    @patch("analyze_noop_ab.trace_for_run")
    def test_first_trace_divergence_reports_first_action_delta(self, trace_for_run_mock):
        trace_for_run_mock.side_effect = [
            [
                {"type": "macro_action", "action_type": "select_map_node", "action_data": {"action": "move_to_map_coord", "col": 1, "row": 2}, "screen_state": {"state_type": "map"}, "state": {"run": {"floor": 1}}},
                {"type": "macro_action", "action_type": "choose_event_option", "action_data": {"action": "choose_event_option", "option_index": 0}, "screen_state": {"state_type": "event"}, "state": {"run": {"floor": 2}}},
            ],
            [
                {"type": "macro_action", "action_type": "select_map_node", "action_data": {"action": "move_to_map_coord", "col": 1, "row": 2}, "screen_state": {"state_type": "map"}, "state": {"run": {"floor": 1}}},
                {"type": "macro_action", "action_type": "choose_event_option", "action_data": {"action": "choose_event_option", "option_index": 1}, "screen_state": {"state_type": "event"}, "state": {"run": {"floor": 2}}},
            ],
        ]

        divergence = first_trace_divergence("baseline", "noop")

        self.assertEqual(divergence["index"], 1)
        self.assertEqual(divergence["baseline_screen"], "event")
        self.assertEqual(divergence["baseline_payload"]["option_index"], 0)
        self.assertEqual(divergence["noop_payload"]["option_index"], 1)

    @patch("analyze_noop_ab.trace_hash")
    @patch("analyze_noop_ab.first_trace_divergence")
    def test_attach_trace_divergences_counts_rows(self, first_trace_mock, trace_hash_mock):
        first_trace_mock.side_effect = [{}, {"index": 3}]
        trace_hash_mock.side_effect = ["a", "a", "b", "c"]
        summary = {
            "per_seed": [
                {"baseline_run_id": "b1", "active_run_id": "n1"},
                {"baseline_run_id": "b2", "active_run_id": "n2"},
            ]
        }

        result = attach_trace_divergences(summary)

        self.assertEqual(result["trace_first_divergence_count"], 1)
        self.assertEqual(result["trace_first_divergence"], {"index": 3})
        self.assertTrue(result["per_seed"][0]["trace_hash_match"])
        self.assertFalse(result["per_seed"][1]["trace_hash_match"])


if __name__ == "__main__":
    unittest.main()
