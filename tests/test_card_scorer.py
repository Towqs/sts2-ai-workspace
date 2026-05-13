import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AI_Training"))

from deck_summary import build_deck_summary
from options.cards import (
    build_card_reward_options,
    enabled_templates,
    load_template_config,
    score_card,
)


def make_large_deck(size=30):
    return [
        {"id": "DEFEND", "type": "Skill", "description": "Gain Block.", "cost": 1}
        for _ in range(size)
    ]


class CardScorerTests(unittest.TestCase):
    def test_default_templates_enable_three_and_disable_self_damage(self):
        config = load_template_config()
        enabled = set(enabled_templates(config))
        self.assertIn("strength_multihit", enabled)
        self.assertIn("barricade_block", enabled)
        self.assertIn("exhaust_engine", enabled)
        self.assertNotIn("self_damage_rupture", enabled)
        self.assertFalse(config["templates"]["self_damage_rupture"]["enabled"])

    def test_card_reward_options_include_cards_and_skip(self):
        state = {
            "run": {"act": 1, "floor": 5},
            "player": {"deck": make_large_deck(24), "deck_size": 24, "hp": 60, "max_hp": 80},
            "card_reward": {
                "can_skip": True,
                "cards": [
                    {"index": 0, "id": "TWIN_STRIKE", "type": "Attack", "cost": 1},
                    {"index": 1, "id": "UNKNOWN_CARD", "type": "Other", "cost": 2},
                    {"index": 2, "id": "SHRUG_IT_OFF", "type": "Skill", "description": "Gain Block. Draw.", "cost": 1},
                ],
            },
        }
        result = build_card_reward_options(state, mode="shadow")
        labels = {option.label for option in result.options}
        self.assertEqual(result.mode, "shadow")
        self.assertIn("choose_card:index_0", labels)
        self.assertIn("choose_card:index_1", labels)
        self.assertIn("choose_card:index_2", labels)
        self.assertIn("skip_reward", labels)
        self.assertEqual(result.to_dict()["legal_option_count"], 4)

    def test_skip_rises_for_bloated_deck_with_weak_candidates(self):
        state = {
            "run": {"act": 2, "floor": 22},
            "player": {"deck": make_large_deck(32), "deck_size": 32, "hp": 50, "max_hp": 80},
            "card_reward": {
                "can_skip": True,
                "cards": [
                    {"index": 0, "id": "UNKNOWN_A", "type": "Other", "cost": 2},
                    {"index": 1, "id": "UNKNOWN_B", "type": "Other", "cost": 2},
                    {"index": 2, "id": "UNKNOWN_C", "type": "Other", "cost": 2},
                ],
            },
        }
        result = build_card_reward_options(state, mode="shadow")
        self.assertEqual(result.selected.label, "skip_reward")
        self.assertGreater(result.selected.score, 0.0)

    def test_multihit_scores_higher_in_strength_template(self):
        state = {
            "run": {"act": 1, "floor": 6},
            "player": {"deck": [{"id": "INFLAME", "type": "Power", "description": "Gain Strength.", "cost": 1}], "deck_size": 1},
        }
        summary = build_deck_summary(state)
        card = {"id": "TWIN_STRIKE", "type": "Attack", "description": "Deal damage twice.", "cost": 1}
        strength = score_card(state, summary, card, template_id="strength_multihit")["score"]
        barricade = score_card(state, summary, card, template_id="barricade_block")["score"]
        self.assertGreater(strength, barricade)


if __name__ == "__main__":
    unittest.main()
