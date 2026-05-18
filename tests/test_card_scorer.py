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
    normalize_card_scorer_mode,
    score_card,
)
from ai_agent import (
    CARD_CANARY_COUNTS,
    SKIPPED_CARD_REWARD_KEYS,
    canary_counter_for_card_reward,
    card_reward_item_key,
    choose_reward_rule_action,
)


def make_large_deck(size=30):
    return [
        {"id": "DEFEND", "type": "Skill", "description": "Gain Block.", "cost": 1}
        for _ in range(size)
    ]


class CardScorerTests(unittest.TestCase):
    def tearDown(self):
        CARD_CANARY_COUNTS.clear()
        SKIPPED_CARD_REWARD_KEYS.clear()

    def test_default_templates_enable_three_and_disable_self_damage(self):
        config = load_template_config()
        enabled = set(enabled_templates(config))
        self.assertIn("strength_multihit", enabled)
        self.assertIn("barricade_block", enabled)
        self.assertIn("exhaust_engine", enabled)
        self.assertNotIn("self_damage_rupture", enabled)
        self.assertFalse(config["templates"]["self_damage_rupture"]["enabled"])
        self.assertEqual(config["template_selection"]["min_consistency_target"], 0.65)
        self.assertEqual(config["active_canary"]["max_active_ratio_per_run"], 0.35)
        self.assertEqual(config["active_canary"]["allow_skip_when_best_card_score_lte"], 0.5)
        self.assertEqual(normalize_card_scorer_mode("active_canary"), "active_canary")
        self.assertEqual(normalize_card_scorer_mode("active_canary_noop"), "active_canary_noop")

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
        first_card = next(option for option in result.to_dict()["options"] if option["label"] == "choose_card:index_0")
        self.assertEqual(first_card["card_id"], "TWIN_STRIKE")
        self.assertIn("score_breakdown", first_card)
        self.assertIn("total_score", first_card)

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
        skip = next(option for option in result.to_dict()["options"] if option["label"] == "skip_reward")
        self.assertIn("skip_diagnostics", skip["metadata"])
        self.assertLess(skip["metadata"]["skip_diagnostics"]["best_card_score"], 0.8)

    def test_skip_competes_with_self_damage_card_in_huge_deck(self):
        state = {
            "run": {"act": 2, "floor": 29},
            "player": {"deck": make_large_deck(32), "deck_size": 32, "hp": 42, "max_hp": 80},
            "card_reward": {
                "can_skip": True,
                "cards": [
                    {"index": 0, "id": "RUPTURE", "name": "Rupture", "type": "Power", "description": "Whenever you lose HP, gain Strength.", "rarity": "Uncommon", "cost": 1},
                    {"index": 1, "id": "JUGGLING", "type": "Power", "rarity": "Uncommon", "cost": 1},
                    {"index": 2, "id": "STAMPEDE", "type": "Power", "rarity": "Uncommon", "cost": 2},
                ],
            },
        }
        result = build_card_reward_options(state, mode="shadow", template_id="strength_multihit")
        labels = [option.label for option in result.options[:2]]
        self.assertIn("skip_reward", labels)

    def test_body_slam_needs_block_support(self):
        state = {
            "run": {"act": 1, "floor": 4},
            "player": {
                "deck": [
                    {"id": "STRIKE", "type": "Attack", "cost": 1},
                    {"id": "STRIKE", "type": "Attack", "cost": 1},
                    {"id": "BASH", "type": "Attack", "cost": 2},
                ],
                "deck_size": 3,
            },
        }
        summary = build_deck_summary(state)
        body_slam = {"id": "BODY_SLAM", "name": "Body Slam", "type": "Attack", "description": "Deal damage equal to your Block.", "cost": 1}
        twin = {"id": "TWIN_STRIKE", "type": "Attack", "description": "Deal damage twice.", "cost": 1}
        body_score = score_card(state, summary, body_slam, template_id="barricade_block")["score"]
        twin_score = score_card(state, summary, twin, template_id="barricade_block")["score"]
        self.assertLessEqual(body_score, twin_score + 0.5)

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

    def test_result_includes_confidence_gap_and_template_lock(self):
        state = {
            "run": {"act": 1, "floor": 5},
            "player": {"deck": [{"id": "INFLAME", "type": "Power", "description": "Gain Strength.", "cost": 1}], "deck_size": 1},
            "_card_template_lock": {"selected_template": "strength_multihit", "locked": True},
            "card_reward": {
                "can_skip": True,
                "cards": [
                    {"index": 0, "id": "TWIN_STRIKE", "name": "Twin Strike", "type": "Attack", "description": "Deal damage twice.", "cost": 1},
                    {"index": 1, "id": "UNKNOWN_CARD", "type": "Other", "cost": 2},
                ],
            },
        }
        result = build_card_reward_options(state, mode="shadow", template_id="strength_multihit")
        payload = result.to_dict()
        self.assertGreaterEqual(payload["confidence_gap"], 0.0)
        self.assertTrue(payload["template_lock"]["locked"])

    def test_canary_counter_counts_unique_card_reward_screens_per_run(self):
        base_state = {
            "_ai_session_id": "run-a",
            "run": {"act": 1, "floor": 5},
            "card_reward": {
                "cards": [
                    {"index": 0, "id": "TWIN_STRIKE"},
                    {"index": 1, "id": "SHRUG_IT_OFF"},
                ]
            },
        }
        _, first, first_new = canary_counter_for_card_reward(base_state)
        _, duplicate, duplicate_new = canary_counter_for_card_reward(base_state)
        changed_screen = {
            **base_state,
            "run": {"act": 1, "floor": 6},
        }
        _, second_screen, second_new = canary_counter_for_card_reward(changed_screen)
        next_run = {
            **base_state,
            "_ai_session_id": "run-b",
        }
        _, next_counter, next_new = canary_counter_for_card_reward(next_run)

        self.assertTrue(first_new)
        self.assertFalse(duplicate_new)
        self.assertTrue(second_new)
        self.assertTrue(next_new)
        self.assertEqual(first["seen"], 2)
        self.assertIs(first, duplicate)
        self.assertIs(first, second_screen)
        self.assertEqual(next_counter["seen"], 1)

    def test_skipped_card_reward_is_not_claimed_again(self):
        state = {
            "_ai_session_id": "run-reward",
            "state_type": "rewards",
            "run": {"act": 2, "floor": 21},
            "rewards": {
                "can_proceed": True,
                "items": [
                    {"index": 0, "type": "potion"},
                    {"index": 1, "type": "card"},
                    {"index": 2, "type": "card"},
                ],
            },
        }
        SKIPPED_CARD_REWARD_KEYS.add(card_reward_item_key(state, state["rewards"]["items"][2], 2))
        payload, info = choose_reward_rule_action(state, "rewards")
        self.assertEqual(payload, {"action": "claim_reward", "index": 0})
        self.assertNotEqual(info["chosen_action"], "claim_reward:index_2:card")

        SKIPPED_CARD_REWARD_KEYS.add(card_reward_item_key(state, state["rewards"]["items"][1], 1))
        state["rewards"]["items"] = state["rewards"]["items"][1:]
        payload, info = choose_reward_rule_action(state, "rewards")
        self.assertEqual(payload, {"action": "proceed"})
        self.assertEqual(info["chosen_action"], "proceed")



if __name__ == "__main__":
    unittest.main()
