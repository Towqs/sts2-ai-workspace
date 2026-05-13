import argparse
import json
import os
import time

import numpy as np


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


def admitted_self_play_scores(workspace_dir):
    path = os.path.join(workspace_dir, "RL_Datasets", "self_play_scores.json")
    data = read_json(path, {"scores": {}})
    scores = data.get("scores") if isinstance(data, dict) else {}
    result = {}
    for run_id, item in (scores or {}).items():
        if not isinstance(item, dict):
            continue
        if item.get("admitted") is not True and item.get("admission_reason") not in ("promising_act1_failure", "boss_progress_failure"):
            continue
        result[str(run_id)] = item
    return result


def load_group_meta(processed_dir):
    path = os.path.join(processed_dir, "candidate_group_meta.json")
    data = read_json(path, {"groups": []})
    groups = data.get("groups") if isinstance(data, dict) else []
    by_group = {}
    for item in groups or []:
        if not isinstance(item, dict):
            continue
        try:
            group_id = int(item.get("group_id"))
        except (TypeError, ValueError):
            continue
        by_group[group_id] = item
    return by_group


def build_rl_subset(x_train, y_train, groups, group_meta, scores):
    selected_mask = np.zeros(len(y_train), dtype=bool)
    row_weights = np.ones(len(y_train), dtype=np.float32)
    selected_groups = set()
    admitted_runs = set()

    for group_id in np.unique(groups):
        meta = group_meta.get(int(group_id), {})
        run_id = str(meta.get("run_id") or "")
        score = scores.get(run_id)
        if not score:
            continue
        mask = groups == group_id
        if not np.any(mask):
            continue
        selected_mask |= mask
        selected_groups.add(int(group_id))
        admitted_runs.add(run_id)
        sample_weight = float(score.get("sample_weight") or 1.0)
        reward = float(score.get("reward") or 0.0)
        progress_bonus = max(0.0, reward) * 0.05
        row_weights[mask] = max(1.0, min(8.0, sample_weight + progress_bonus))

    return (
        x_train[selected_mask],
        y_train[selected_mask],
        groups[selected_mask],
        row_weights[selected_mask],
        selected_groups,
        admitted_runs,
    )


def train(processed_dir, workspace_dir, epochs=30, min_groups=5, lr=3e-4):
    x_path = os.path.join(processed_dir, "candidate_X_train.npy")
    y_path = os.path.join(processed_dir, "candidate_Y_train.npy")
    group_path = os.path.join(processed_dir, "candidate_group_train.npy")
    bc_model_path = os.path.join(processed_dir, "candidate_bc_model_best.pth")
    rl_model_path = os.path.join(processed_dir, "candidate_rl_model_best.pth")
    rl_metadata_path = os.path.join(processed_dir, "candidate_rl_metadata.json")

    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(group_path)):
        print("RL fine-tune skipped: candidate dataset files are missing. Run data_pipeline.py first.")
        return 0
    if not os.path.exists(bc_model_path):
        print("RL fine-tune skipped: candidate_bc_model_best.pth is missing. Run train_candidate_bc.py first.")
        return 0

    scores = admitted_self_play_scores(workspace_dir)
    if not scores:
        print("RL fine-tune skipped: no admitted self-play runs yet.")
        write_json(rl_metadata_path, {
            "status": "skipped",
            "reason": "no_admitted_self_play_runs",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return 0

    group_meta = load_group_meta(processed_dir)
    if not group_meta:
        print("RL fine-tune skipped: candidate_group_meta.json is missing or empty. Re-run data_pipeline.py.")
        write_json(rl_metadata_path, {
            "status": "skipped",
            "reason": "missing_candidate_group_meta",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return 0

    x_train = np.load(x_path)
    y_train = np.load(y_path).astype(np.float32)
    groups = np.load(group_path)

    x_rl, y_rl, groups_rl, weights_rl, selected_groups, admitted_runs = build_rl_subset(
        x_train,
        y_train,
        groups,
        group_meta,
        scores,
    )
    positives = int(y_rl.sum()) if len(y_rl) else 0
    if len(selected_groups) < min_groups or positives <= 0:
        print(
            "RL fine-tune skipped: not enough admitted candidate groups "
            f"({len(selected_groups)}/{min_groups}, positives={positives})."
        )
        write_json(rl_metadata_path, {
            "status": "skipped",
            "reason": "not_enough_admitted_candidate_groups",
            "admitted_runs": sorted(admitted_runs),
            "admitted_groups": int(len(selected_groups)),
            "positives": positives,
            "min_groups": int(min_groups),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return 0

    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from train_candidate_bc import CandidateBCScorer, group_top1_accuracy

    input_dim = int(x_train.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateBCScorer(input_dim).to(device)
    model.load_state_dict(torch.load(bc_model_path, map_location=device, weights_only=True))

    x_tensor = torch.tensor(x_rl, dtype=torch.float32)
    y_tensor = torch.tensor(y_rl, dtype=torch.float32)
    w_tensor = torch.tensor(weights_rl, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor, w_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_top1 = 0.0
    started = time.time()
    print(
        "Starting Phase 2 RL-style reward-weighted fine-tune: "
        f"rows={len(y_rl)}, groups={len(selected_groups)}, admitted_runs={len(admitted_runs)}, device={device}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y, batch_w in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            per_row = F.binary_cross_entropy_with_logits(logits, batch_y, reduction="none")
            loss = (per_row * batch_w).sum() / batch_w.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(dataloader), 1)
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                logits = model(x_tensor.to(device)).detach().cpu().numpy()
            top1 = group_top1_accuracy(logits, y_rl, groups_rl)
            print(f"Epoch {epoch}/{epochs} | weighted_loss={avg_loss:.4f} | admitted_group_top1={top1:.2f}%")
        else:
            top1 = best_top1
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_top1 = max(best_top1, top1)
            torch.save(model.state_dict(), rl_model_path)

    metadata = {
        "status": "ok",
        "algorithm": "reward_weighted_candidate_policy_finetune",
        "base_model": "candidate_bc_model_best.pth",
        "model_path": "candidate_rl_model_best.pth",
        "samples": int(len(y_rl)),
        "groups": int(len(selected_groups)),
        "positives": int(positives),
        "features": int(input_dim),
        "admitted_runs": sorted(admitted_runs),
        "admitted_run_count": int(len(admitted_runs)),
        "epochs": int(epochs),
        "best_loss": float(best_loss),
        "best_group_top1": float(best_top1),
        "device": str(device),
        "elapsed_sec": round(time.time() - started, 2),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(rl_metadata_path, metadata)
    print(f"RL fine-tune complete. Saved {rl_model_path}")
    return 0


def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(workspace_dir, "AI_Training", "ProcessedParams")
    parser = argparse.ArgumentParser(description="Phase 2 reward-weighted fine-tune for the candidate combat policy.")
    parser.add_argument("--processed-dir", default=processed_dir)
    parser.add_argument("--workspace-dir", default=workspace_dir)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-groups", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    return train(args.processed_dir, args.workspace_dir, epochs=args.epochs, min_groups=args.min_groups, lr=args.lr)


if __name__ == "__main__":
    raise SystemExit(main())
