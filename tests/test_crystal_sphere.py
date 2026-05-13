import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AI_Training"))

from ai_agent import CARD_TEMPLATE_LOCKS, choose_macro_action, locked_template_for_card_reward


class CrystalSphereRuleTests(unittest.TestCase):
    def test_can_proceed_wins_even_in_explore_mode(self):
        state = {
            "state_type": "crystal_sphere",
            "crystal_sphere": {
                "can_proceed": True,
                "divinations_left_text": "还剩下0次占卜。",
                "cells": [
                    {"x": 3, "y": 0, "is_hidden": True, "is_clickable": True},
                ],
            },
        }
        payload, info = choose_macro_action(
            macro_agent={"sentinel": True},
            state=state,
            exploration={"constraint_mode": "explore"},
        )
        self.assertEqual(payload, {"action": "crystal_sphere_proceed"})
        self.assertEqual(info["chosen_action"], "crystal_sphere_proceed")

    def test_cells_field_is_used_for_clickable_cells(self):
        state = {
            "state_type": "crystal_sphere",
            "crystal_sphere": {
                "can_proceed": False,
                "divinations_left_text": "还剩下2次占卜。",
                "tool": "big",
                "grid_width": 11,
                "grid_height": 11,
                "cells": [
                    {"x": 0, "y": 0, "is_hidden": True, "is_clickable": True},
                    {"x": 5, "y": 5, "is_hidden": True, "is_clickable": True},
                ],
            },
        }
        payload, info = choose_macro_action(
            macro_agent={"sentinel": True},
            state=state,
            exploration={"constraint_mode": "explore"},
        )
        self.assertEqual(payload, {"action": "crystal_sphere_click_cell", "x": 5, "y": 5})
        self.assertEqual(info["chosen_action"], "crystal_sphere_click_cell:5,5")


class TemplateLockTests(unittest.TestCase):
    def test_template_locks_after_warmup(self):
        CARD_TEMPLATE_LOCKS.clear()
        state = {
            "run": {"act": 1, "floor": 5},
            "player": {
                "deck": [
                    {"id": "INFLAME", "type": "Power", "description": "Gain Strength.", "cost": 1},
                    {"id": "TWIN_STRIKE", "type": "Attack", "description": "Deal damage twice.", "cost": 1},
                ],
                "deck_size": 2,
            },
            "card_reward": {
                "can_skip": True,
                "cards": [
                    {"index": 0, "id": "TWIN_STRIKE", "type": "Attack", "description": "Deal damage twice.", "cost": 1},
                ],
            },
        }
        for floor in (5, 6, 7):
            state["run"]["floor"] = floor
            template_id, _summary, lock = locked_template_for_card_reward(state, session_id="test_lock")
        self.assertEqual(template_id, lock["locked_template"])
        self.assertTrue(lock["locked"])
        self.assertEqual(lock["card_reward_count"], 3)


if __name__ == "__main__":
    unittest.main()
