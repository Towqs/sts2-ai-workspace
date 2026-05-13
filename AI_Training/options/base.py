from dataclasses import dataclass, field


OPTION_SCHEMA_VERSION = "option_schema_v1"
OPTION_FEATURES_VERSION = "option_features_v1"


@dataclass
class Option:
    label: str
    payload: dict
    kind: str = "macro"
    score: float = 0.0
    reasons: list = field(default_factory=list)
    features: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    index: int = -1

    def to_dict(self, include_features=False):
        item = {
            "label": self.label,
            "payload": self.payload,
            "kind": self.kind,
            "score": round(float(self.score), 4),
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
            "index": int(self.index),
        }
        if include_features:
            item["features"] = list(self.features)
        return item


@dataclass
class OptionResult:
    options: list
    selected: Option | None = None
    mode: str = "shadow"
    template_id: str = ""
    option_schema: str = OPTION_SCHEMA_VERSION
    option_features_version: str = OPTION_FEATURES_VERSION
    state_features_version: str = ""
    deck_summary: dict = field(default_factory=dict)
    archetype_consistency: dict = field(default_factory=dict)

    def to_dict(self, include_features=False):
        return {
            "mode": self.mode,
            "template_id": self.template_id,
            "option_schema": self.option_schema,
            "option_features_version": self.option_features_version,
            "state_features_version": self.state_features_version,
            "legal_option_count": len(self.options),
            "selected": self.selected.to_dict(include_features=include_features) if self.selected else None,
            "options": [option.to_dict(include_features=include_features) for option in self.options],
            "deck_summary": dict(self.deck_summary),
            "archetype_consistency": dict(self.archetype_consistency),
        }


def ranked_options(options):
    return sorted(options, key=lambda option: float(option.score), reverse=True)
