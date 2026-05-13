import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AI_Training"))

from ai_agent import choose_macro_action


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


if __name__ == "__main__":
    unittest.main()
