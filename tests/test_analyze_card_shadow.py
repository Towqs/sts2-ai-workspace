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
        self.since_ms = kwargs.get("since_ms", 0)
        self.new_logic_only = kwargs.get("new_logic_only", False)


class AnalyzeCardShadowTests(unittest.TestCase):
    def test_metrics_and_report_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "card_scorer_2026-05-13.jsonl"
            log_path.write_text(
                "\n".join([
                    '{"type":"card_scorer_shadow","run_id":"r1","floor":3,"legacy_chosen_action":"choose_card:index_0","recommended_action":"choose_card:index_0","legal_option_count":4,"template_id":"strength_multihit","template_locked":true,"locked_template":"strength_multihit","skip_score":-1.0,"best_card_score":2.0,"archetype_consistency":{"consistency":0.8},"deck_summary":{"deck_size":12,"bloat_score":0.0},"reward_terms":{"deck_fit":0.4},"options":[{"label":"choose_card:index_0","score":2.0},{"label":"choose_card:index_1","score":1.2},{"label":"skip_reward","score":-1.0}],"selected":{"label":"choose_card:index_0","score":2.0}}',
                    '{"type":"card_scorer_shadow","run_id":"r1","floor":5,"legacy_chosen_action":"choose_card:index_1","recommended_action":"skip_reward","raw_scorer_action":"choose_card:index_3","extra_card_index_fallback":true,"legal_option_count":4,"template_id":"strength_multihit","template_locked":true,"locked_template":"strength_multihit","skip_score":2.5,"best_card_score":0.1,"archetype_consistency":{"consistency":0.6},"deck_summary":{"deck_size":31,"bloat_score":0.5},"reward_terms":{"deck_fit":0.8,"bad":"Infinity"},"options":[{"label":"skip_reward","score":2.5},{"label":"choose_card:index_1","score":0.1},{"label":"choose_card:index_2","score":"NaN"},{"label":"choose_card:index_0","score":"Infinity"}],"selected":{"label":"skip_reward","score":2.5}}',
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
            self.assertEqual(metrics["effective_fallback_count"], 1)
            self.assertEqual(metrics["extra_card_index_fallback_count"], 1)
            self.assertEqual(summary["raw_scorer_distribution"]["choose_card:index_3"], 1)
            self.assertTrue(math.isclose(metrics["template_locked_rate"], 1.0))
            self.assertTrue(math.isclose(metrics["template_sequence_consistency"], 1.0))
            self.assertEqual(metrics["locked_template_distribution"]["strength_multihit"], 2)
            self.assertTrue(math.isclose(metrics["avg_skip_score"], 0.75))
            self.assertTrue(Path(summary["report_path"]).exists())

    def test_since_ms_filters_old_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "card_scorer_2026-05-13.jsonl"
            log_path.write_text(
                "\n".join([
                    '{"type":"card_scorer_shadow","timestamp":1000,"run_id":"old","legacy_chosen_action":"choose_card:index_0","recommended_action":"choose_card:index_0","legal_option_count":4,"options":[{"label":"choose_card:index_0","score":1.0},{"label":"skip_reward","score":0.0}]}',
                    '{"type":"card_scorer_shadow","timestamp":2000,"run_id":"new","legacy_chosen_action":"choose_card:index_0","recommended_action":"skip_reward","legal_option_count":4,"options":[{"label":"choose_card:index_0","score":0.0},{"label":"skip_reward","score":1.0}]}',
                ]),
                encoding="utf-8",
            )
            summary = analyze(Args(files=[str(log_path)], since_ms=1500))
            metrics = summary["metrics"]
            self.assertEqual(metrics["total_card_reward_events"], 1)
            self.assertEqual(metrics["run_count"], 1)
            self.assertEqual(metrics["scorer_disagreed_with_old_policy"], 1)

    def test_new_logic_only_filters_legacy_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "card_scorer_2026-05-13.jsonl"
            log_path.write_text(
                "\n".join([
                    '{"type":"card_scorer_shadow","run_id":"legacy","legacy_chosen_action":"choose_card:index_0","recommended_action":"choose_card:index_0","legal_option_count":4,"options":[{"label":"choose_card:index_0","score":1.0},{"label":"skip_reward","score":0.0}]}',
                    '{"type":"card_scorer_shadow","run_id":"new","scorer_logic_version":"ironclad_card_scorer_logic_v1_5b","template_logic_version":"ironclad_template_lock_v1","skip_logic_version":"ironclad_skip_logic_v1_5b","legacy_chosen_action":"choose_card:index_0","recommended_action":"skip_reward","legal_option_count":4,"options":[{"label":"choose_card:index_0","score":0.0},{"label":"skip_reward","score":1.0}]}',
                ]),
                encoding="utf-8",
            )
            summary = analyze(Args(files=[str(log_path)], new_logic_only=True))
            metrics = summary["metrics"]
            self.assertEqual(metrics["report_scope"], "new_logic_only")
            self.assertEqual(metrics["total_card_reward_events"], 1)
            self.assertEqual(metrics["run_count"], 1)
            self.assertEqual(metrics["scorer_logic_versions"]["ironclad_card_scorer_logic_v1_5b"], 1)


if __name__ == "__main__":
    unittest.main()
