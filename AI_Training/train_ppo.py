import argparse
import glob
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ppo_policy import PPO_INPUT_DIM, build_ppo_model, ppo_paths, read_json, write_json


def iter_rollout_records(ppo_dir):
    for path in sorted(glob.glob(os.path.join(ppo_dir, "ppo_rollouts_*.jsonl"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if item.get("type") == "ppo_step":
                        item["_file"] = path
                        yield item
        except FileNotFoundError:
            continue


def terminal_reward(score):
    if not isinstance(score, dict):
        return 0.0
    reward = 0.0
    max_act = int(score.get("max_act") or 0)
    max_floor = int(score.get("max_floor") or 0)
    if max_act >= 2 or score.get("reason_group") == "clear":
        reward += 3.0
    if score.get("reason_group") == "death":
        reward -= 2.0
        if max_act <= 1 and max_floor < 10:
            reward -= 1.0
        if max_act <= 1 and max_floor < 6:
            reward -= 1.0
    if score.get("stuck"):
        reward -= 1.5
    reward += min(1.5, float(score.get("boss_damage") or 0.0) * 0.02)
    return max(-4.0, min(4.0, reward))


def load_steps(workspace_dir, max_records=50000):
    ppo_dir = os.path.join(workspace_dir, "RL_Datasets", "PPO")
    scores = read_json(os.path.join(workspace_dir, "RL_Datasets", "self_play_scores.json"), {"scores": {}}).get("scores", {})
    by_run = defaultdict(list)
    for item in iter_rollout_records(ppo_dir):
        rows = item.get("features") or []
        chosen = item.get("chosen_index")
        if not rows or chosen is None:
            continue
        if not (0 <= int(chosen) < len(rows)):
            continue
        try:
            reward = float(item.get("reward") or 0.0)
            logprob = float(item.get("logprob") or 0.0)
            value = float(item.get("value") or 0.0)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(reward) and np.isfinite(logprob) and np.isfinite(value)):
            continue
        if item.get("reward_schema") != "ironclad_v1" and abs(reward) > 1.001:
            continue
        by_run[str(item.get("run_id") or "")].append(item)
    all_steps = []
    for run_id, steps in by_run.items():
        steps.sort(key=lambda x: int(x.get("timestamp") or 0))
        if steps:
            steps[-1]["reward"] = float(steps[-1].get("reward") or 0.0) + terminal_reward(scores.get(run_id))
            steps[-1]["done"] = True
        all_steps.extend(steps)
    all_steps.sort(key=lambda x: int(x.get("timestamp") or 0))
    if max_records and len(all_steps) > max_records:
        all_steps = all_steps[-max_records:]
    schema_steps = [step for step in all_steps if step.get("reward_schema") == "ironclad_v1"]
    if len(schema_steps) >= 256:
        return schema_steps
    return all_steps


def compute_advantages(steps, gamma=0.99, lam=0.95):
    by_run = defaultdict(list)
    for idx, step in enumerate(steps):
        by_run[str(step.get("run_id") or "")].append(idx)
    returns = np.zeros(len(steps), dtype=np.float32)
    advantages = np.zeros(len(steps), dtype=np.float32)
    for _run_id, indices in by_run.items():
        next_adv = 0.0
        next_value = 0.0
        for idx in reversed(indices):
            step = steps[idx]
            reward = float(step.get("reward") or 0.0)
            value = float(step.get("value") or 0.0)
            done = bool(step.get("done"))
            if done:
                next_value = 0.0
                next_adv = 0.0
            delta = reward + gamma * next_value * (0.0 if done else 1.0) - value
            next_adv = delta + gamma * lam * (0.0 if done else 1.0) * next_adv
            advantages[idx] = next_adv
            returns[idx] = next_adv + value
            next_value = value
    if len(advantages):
        std = float(advantages.std())
        if std > 1e-6:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
    return returns, advantages


def step_tensors(step, device):
    rows = np.asarray(step.get("features") or [], dtype=np.float32)
    if rows.ndim != 2:
        rows = rows.reshape(1, PPO_INPUT_DIM)
    if rows.shape[1] != PPO_INPUT_DIM:
        if rows.shape[1] > PPO_INPUT_DIM:
            rows = rows[:, :PPO_INPUT_DIM]
        else:
            rows = np.pad(rows, ((0, 0), (0, PPO_INPUT_DIM - rows.shape[1])), mode="constant")
    return torch.tensor(rows, dtype=torch.float32, device=device), int(step.get("chosen_index") or 0)


def init_from_candidate_scorer(model, workspace_dir, device):
    processed_dir = os.path.join(workspace_dir, "AI_Training", "ProcessedParams")
    sources = [
        ("candidate_rl", os.path.join(processed_dir, "candidate_rl_model_best.pth")),
        ("candidate_bc", os.path.join(processed_dir, "candidate_bc_model_best.pth")),
    ]
    for name, path in sources:
        if not os.path.exists(path):
            continue
        try:
            state = torch.load(path, map_location=device, weights_only=True)
            with torch.no_grad():
                copies = [
                    ("net.0.weight", model.body[0].weight),
                    ("net.0.bias", model.body[0].bias),
                    ("net.3.weight", model.body[2].weight),
                    ("net.3.bias", model.body[2].bias),
                    ("net.5.weight", model.policy_head.weight),
                    ("net.5.bias", model.policy_head.bias),
                ]
                for key, target in copies:
                    if key not in state:
                        continue
                    source = state[key].to(device)
                    if source.ndim == 2 and target.ndim == 2:
                        rows = min(source.shape[0], target.shape[0])
                        cols = min(source.shape[1], target.shape[1])
                        target[:rows, :cols].copy_(source[:rows, :cols])
                    elif source.ndim == 1 and target.ndim == 1:
                        rows = min(source.shape[0], target.shape[0])
                        target[:rows].copy_(source[:rows])
            return name
        except Exception:
            continue
    return ""


def train(processed_dir, workspace_dir, epochs=4, batch_size=256, lr=3e-4, clip=0.2, value_coef=0.5, entropy_coef=0.01):
    paths = ppo_paths(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    steps = load_steps(workspace_dir)
    if len(steps) < 20:
        write_json(paths["metadata"], {
            "status": "skipped",
            "reason": "not_enough_ppo_steps",
            "steps": len(steps),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"PPO skipped: only {len(steps)} rollout steps.")
        return 0

    returns, advantages = compute_advantages(steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ppo_model(device)
    init_source = ""
    if os.path.exists(paths["latest"]):
        try:
            model.load_state_dict(torch.load(paths["latest"], map_location=device, weights_only=True))
        except Exception:
            pass
    else:
        init_source = init_from_candidate_scorer(model, workspace_dir, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    indices = np.arange(len(steps))
    started = time.time()
    last_metrics = {}

    for _epoch in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            losses = []
            policy_losses = []
            value_losses = []
            entropies = []
            for idx in batch:
                features, chosen = step_tensors(steps[int(idx)], device)
                logits, values = model(features)
                log_probs = F.log_softmax(logits, dim=0)
                probs = torch.softmax(logits, dim=0)
                chosen = max(0, min(chosen, int(logits.numel()) - 1))
                new_logprob = log_probs[chosen]
                old_logprob = torch.tensor(float(steps[int(idx)].get("logprob") or 0.0), dtype=torch.float32, device=device)
                ratio = torch.exp(new_logprob - old_logprob)
                adv = torch.tensor(float(advantages[int(idx)]), dtype=torch.float32, device=device)
                ret = torch.tensor(float(returns[int(idx)]), dtype=torch.float32, device=device)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
                policy_loss = -torch.min(unclipped, clipped)
                value_loss = F.mse_loss(values[chosen], ret)
                entropy = -(probs * log_probs).sum()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                losses.append(loss)
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropies.append(entropy.detach())
            if not losses:
                continue
            optimizer.zero_grad()
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_metrics = {
                "loss": float(total_loss.detach().cpu()),
                "policy_loss": float(torch.stack(policy_losses).mean().cpu()),
                "value_loss": float(torch.stack(value_losses).mean().cpu()),
                "entropy": float(torch.stack(entropies).mean().cpu()),
            }

    torch.save(model.state_dict(), paths["latest"])
    scores = read_json(os.path.join(workspace_dir, "RL_Datasets", "self_play_scores.json"), {"scores": {}}).get("scores", {})
    recent_scores = [v for v in scores.values() if isinstance(v, dict)]
    recent_scores.sort(key=lambda x: str(x.get("updated_at") or ""))
    control = read_json(os.path.join(workspace_dir, "AI_Training", "control_state.json"), {})
    fixed_seed = str(control.get("ppo_fixed_seed") or "101").strip().upper()
    fixed_scores = [x for x in recent_scores if str(x.get("seed") or "").strip().upper() == fixed_seed]
    metric_scores = (fixed_scores[-20:] if fixed_scores else recent_scores[-20:])
    recent_scores = recent_scores[-20:]
    avg_floor = float(np.mean([float(x.get("max_floor") or 0.0) for x in metric_scores])) if metric_scores else 0.0
    avg_boss_damage = float(np.mean([float(x.get("boss_damage") or 0.0) for x in metric_scores])) if metric_scores else 0.0
    act1_clears = sum(1 for x in metric_scores if int(x.get("max_act") or 0) >= 2 or x.get("reason_group") == "clear")
    death_rate = float(np.mean([1.0 if x.get("reason_group") == "death" else 0.0 for x in metric_scores])) if metric_scores else 0.0
    early_death_rate = float(np.mean([
        1.0 if x.get("reason_group") == "death" and int(x.get("max_act") or 0) <= 1 and int(x.get("max_floor") or 0) < 10 else 0.0
        for x in metric_scores
    ])) if metric_scores else 0.0
    old_meta = read_json(paths["metadata"], {})
    old_best = float(old_meta.get("best_metric") or -1e9)
    metric = avg_floor + avg_boss_damage * 0.05 + act1_clears * 25.0 - death_rate * 4.0 - early_death_rate * 8.0
    best_updated = metric >= old_best
    if best_updated:
        torch.save(model.state_dict(), paths["best"])
    if act1_clears > 0:
        torch.save(model.state_dict(), paths["clear"])
    metadata = {
        "status": "ok",
        "algorithm": "ppo_v0_candidate_policy",
        "model_version": f"ppo:s{len(steps)}:{time.strftime('%Y%m%d%H%M%S')}",
        "steps": int(len(steps)),
        "reward_schema": "ironclad_v1",
        "schema_steps": int(sum(1 for step in steps if step.get("reward_schema") == "ironclad_v1")),
        "runs": int(len({str(s.get("run_id") or "") for s in steps})),
        "input_dim": PPO_INPUT_DIM,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "clip": float(clip),
        "gamma": 0.99,
        "lambda": 0.95,
        "latest_path": "ppo_policy_latest.pth",
        "best_path": "ppo_policy_best.pth" if best_updated or os.path.exists(paths["best"]) else "",
        "clear_path": "ppo_policy_clear.pth" if act1_clears > 0 or os.path.exists(paths["clear"]) else "",
        "init_source": init_source,
        "fixed_seed": fixed_seed,
        "metric_scope": "fixed_seed" if fixed_scores else "recent_all",
        "best_metric": float(max(metric, old_best)),
        "latest_metric": float(metric),
        "avg_floor_20": float(avg_floor),
        "avg_boss_damage_20": float(avg_boss_damage),
        "act1_clears_20": int(act1_clears),
        "death_rate_20": float(death_rate),
        "early_death_rate_20": float(early_death_rate),
        "best_updated": bool(best_updated),
        "metrics": last_metrics,
        "device": str(device),
        "elapsed_sec": round(time.time() - started, 2),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(paths["metadata"], metadata)
    print(f"PPO training complete. steps={len(steps)} latest={paths['latest']}")
    return 0


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(workspace_dir, "AI_Training", "ProcessedPPOParams")
    parser = argparse.ArgumentParser(description="Train PPO v0 policy from STS2 self-play rollouts.")
    parser.add_argument("--processed-dir", default=processed_dir)
    parser.add_argument("--workspace-dir", default=workspace_dir)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    return train(args.processed_dir, args.workspace_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    raise SystemExit(main())
