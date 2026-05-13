import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AI_Training"))

from analyze_card_shadow import analyze


class Args:
    def __init__(self, **kwargs):
        self.date = kwargs.get("date", "2026-05-13")
        self.all = kwargs.get("all", False)
        self.files = kwargs.get("files")
        self.log_dir = kwargs.get("log_dir", "")
        self.report = kwargs.get("report", "")
        self.report_dir = kwargs.get("report_dir", "")


class AnalyzeCardShadowTests(unittest.TestCase):
    def test_metrics_and_report_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "card_scorer_2026-05-13.jsonl"
            log_path.write_text(
                "\n".join([
                    '{"type":"card_scorer_shadow","run_id":"r1","floor":3,"legacy_chosen_action":"choose_card:index_0","recommended_action":"choose_card:index_0","legal_option_count":4,"template_id":"strength_multihit","archetype_consistency":{"consistency":0.8},"deck_summary":{"deck_size":12,"bloat_score":0.0},"reward_terms":{"deck_fit":0.4},"options":[{"label":"choose_card:index_0","score":2.0},{"label":"choose_card:index_1","score":1.2},{"label":"skip_reward","score":-1.0}],"selected":{"label":"choose_card:index_0","score":2.0}}',
                    '{"type":"card_scorer_shadow","run_id":"r1","floor":5,"legacy_chosen_action":"choose_card:index_1","recommended_action":"skip_reward","legal_option_count":4,"template_id":"exhaust_engine","archetype_consistency":{"consistency":0.6},"deck_summary":{"deck_size":31,"bloat_score":0.5},"reward_terms":{"deck_fit":0.8,"bad":"Infinity"},"options":[{"label":"skip_reward","score":2.5},{"label":"choose_card:index_1","score":0.1},{"label":"choose_card:index_2","score":"NaN"},{"label":"choose_card:index_0","score":"Infinity"}],"selected":{"label":"skip_reward","score":2.5}}',
                ]),
                encoding="utf-8",
            )
            report_dir = tmp_path / "reports"
            summary = analyze(Args(files=[str(log_path)], report="auto", report_dir=str(report_dir)))
            metrics = summary["metrics"]
            self.assertEqual(metrics["total_card_reward_events"], 2)
            self.assertEqual(metrics["scorer_disagreed_with_old_policy"], 1)
            self.assertTrue(math.isclose(metrics["old_vs_scorer_agreement_rate"], 0.5))
            self.assertTrue(math.isclose(metrics["scorer_recommended_skip_rate"], 0.5))
            self.assertEqual(metrics["score_nan_count"], 1)
            self.assertEqual(metrics["score_inf_count"], 1)
            self.assertEqual(metrics["reward_term_inf_count"], 1)
            self.assertEqual(metrics["reward_term_distribution"]["deck_fit"]["count"], 2)
            self.assertTrue(Path(summary["report_path"]).exists())


if __name__ == "__main__":
    unittest.main()
