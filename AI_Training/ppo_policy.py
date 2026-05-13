import json
import os
import time

import torch
import torch.nn as nn


PPO_INPUT_DIM = 160


class PPOPolicy(nn.Module):
    def __init__(self, input_dim=PPO_INPUT_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.body = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, 1)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        hidden = self.body(x)
        return self.policy_head(hidden).squeeze(-1), self.value_head(hidden).squeeze(-1)


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ppo_paths(processed_dir):
    return {
        "dir": processed_dir,
        "latest": os.path.join(processed_dir, "ppo_policy_latest.pth"),
        "best": os.path.join(processed_dir, "ppo_policy_best.pth"),
        "clear": os.path.join(processed_dir, "ppo_policy_clear.pth"),
        "metadata": os.path.join(processed_dir, "ppo_metadata.json"),
    }


def build_ppo_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PPOPolicy(PPO_INPUT_DIM).to(device)


def load_ppo_policy(processed_dir, mode="ppo_experiment", allow_untrained=False, device=None):
    paths = ppo_paths(processed_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ppo_model(device)
    metadata = read_json(paths["metadata"], {})
    wanted = paths["best"] if str(mode) == "ppo_best" else paths["latest"]
    source = "best" if wanted == paths["best"] else "latest"
    if not os.path.exists(wanted):
        if allow_untrained:
            model.eval()
            return {
                "model": model,
                "device": device,
                "input_dim": PPO_INPUT_DIM,
                "metadata": {"status": "untrained", "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                "model_version": "ppo:untrained",
                "trained": False,
                "source": "untrained",
            }
        else:
            return None
    try:
        state_dict = torch.load(wanted, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        version = metadata.get("model_version") or f"ppo:{source}:{int(os.path.getmtime(wanted))}"
        return {
            "model": model,
            "device": device,
            "input_dim": PPO_INPUT_DIM,
            "metadata": metadata,
            "model_version": version,
            "trained": True,
            "source": source,
        }
    except Exception as exc:
        if allow_untrained:
            model.eval()
            return {
                "model": model,
                "device": device,
                "input_dim": PPO_INPUT_DIM,
                "metadata": {"status": "load_failed", "error": str(exc)},
                "model_version": "ppo:untrained",
                "trained": False,
                "source": "untrained_after_error",
            }
        return None
