import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AI_Training"))

from deck_summary import build_deck_summary, classify_card


class DeckSummaryTests(unittest.TestCase):
    def test_empty_state_is_safe(self):
        summary = build_deck_summary({})
        self.assertEqual(summary["deck_size"], 0)
        self.assertEqual(summary["attack_count"], 0)
        self.assertEqual(summary["skill_count"], 0)
        self.assertTrue(math.isfinite(summary["bloat_score"]))

    def test_mixed_english_and_chinese_cards_are_counted(self):
        state = {
            "player": {
                "deck": [
                    {"id": "STRIKE", "name": "\u6253\u51fb", "type": "Attack", "cost": 1},
                    {"id": "BATTLE_TRANCE", "name": "Battle Trance", "type": "Skill", "cost": 0},
                    {"name": "\u672a\u77e5\u6280\u80fd", "type": "\u6280\u80fd", "cost": 2},
                    {"id": "INFLAME", "type": "Power", "description": "Gain Strength.", "cost": 1},
                ],
                "deck_size": 4,
            }
        }
        summary = build_deck_summary(state)
        self.assertEqual(summary["deck_size"], 4)
        self.assertEqual(summary["attack_count"], 1)
        self.assertEqual(summary["skill_count"], 2)
        self.assertEqual(summary["power_count"], 1)
        self.assertGreaterEqual(summary["draw_count"], 1)
        self.assertGreaterEqual(summary["strength_sources"], 1)

    def test_unknown_card_does_not_crash_classifier(self):
        tags = classify_card({"name": "\u795e\u79d8\u724c", "type": "\u672a\u77e5", "cost": "?"})
        self.assertEqual(tags["type"], "other")
        self.assertEqual(tags["cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
