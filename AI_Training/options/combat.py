from combat_actions import enumerate_combat_actions
from options.base import OPTION_FEATURES_VERSION, OPTION_SCHEMA_VERSION, Option


COMBAT_OPTION_KIND = "combat"


def combat_candidate_to_option(candidate, index=0):
    return Option(
        label=getattr(candidate, "label", ""),
        payload=getattr(candidate, "payload", {}),
        kind=getattr(candidate, "kind", COMBAT_OPTION_KIND),
        score=0.0,
        reasons=[],
        features=list(getattr(candidate, "features", []) or []),
        metadata={
            "option_schema": OPTION_SCHEMA_VERSION,
            "option_features_version": OPTION_FEATURES_VERSION,
            "card_id": getattr(candidate, "card_id", ""),
            "card_index": getattr(candidate, "card_index", -1),
            "target_id": getattr(candidate, "target_id", ""),
            "potion_id": getattr(candidate, "potion_id", ""),
        },
        index=index,
    )


def enumerate_combat_options(state, include_end_turn=True, include_potions=True):
    candidates = enumerate_combat_actions(
        state,
        include_end_turn=include_end_turn,
        include_potions=include_potions,
    )
    return [combat_candidate_to_option(candidate, idx) for idx, candidate in enumerate(candidates)]
