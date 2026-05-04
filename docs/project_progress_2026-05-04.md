# Project Progress: 2026-05-04

## Summary

项目已经从“能否跑起来”进入“能否稳定产出可用自训练数据”的阶段。控制面板、自训练 runner、数据采集、训练管线和 ModelZoo 快照闭环已经存在；下一阶段重点是修稳数据完整性、self-play 状态判定，并规划固定 seed / seed 池能力。

## Completed

- 控制面板可以管理 AI 托管、自训练、训练、模型切换和数据导出。
- `self_play_runner.py` 可以启动自训练循环、开新局、监控 run、评分并按入训批次触发训练。
- AI 战斗数据和宏观数据已经写入 `RL_Datasets`。
- 训练管线已经生成战斗模型、候选动作模型和宏观模型。
- ModelZoo 已经保存自动训练快照和人工基线快照。
- `run_summary.py` 和控制面板现在能展示更清晰的 self-play 入训/拒绝原因、阶段标签和最近趋势。

## Current State

- 当前分支：`main`
- 远端：`origin/main`
- 当前激活模型：`auto_train_20260503_001123_984d27`
- 最新训练产物更新时间：2026-05-03
- 最新训练产物：
  - `AI_Training/ProcessedParams/bc_model_best.pth`
  - `AI_Training/ProcessedParams/candidate_bc_model_best.pth`
  - `AI_Training/ProcessedMacroParams/macro_bc_model_best.pth`
- 最近 self-play 能启动并采集数据，但还没有产出入训 run。
- 最近 self-play run `ai_20260504_135614_8a83b887` 到达 Act 1 / floor 17，按当前规则被判定为早期失败，未入训。
- `RL_Datasets/self_play_scores.json` 目前记录了 2 条 self-play 评分，均未入训。

## Current Risks

- self-play 结束状态曾显示为 `error`，但 message 像是启动流程中的成功提示，需要继续修正状态判定。
- 最近 self-play 的 `data_health` 曾出现 `missing`，说明 run summary 或数据 flush 判断还不够稳。
- 固定 seed 目前只是计划，尚未实现；C# 游戏 API 当前仍拒绝非空 seed。
- 如果直接大规模自训练，可能会积累大量无法入训或数据不完整的 run。

## Next Stage

优先目标不是立刻扩大训练规模，而是让自训练稳定产出可用数据：

1. 修复 self-play 结束状态误报和 `data_health=missing` 的根因。
2. 确认 run 数据 flush、summary、评分和入训标记一致。
3. 落地单个固定 seed，从 UI、Python runner 到 C# 游戏 API 全链路打通。
4. 将 seed 写入每条 run 数据和 self-play 评分。
5. 确认同 seed 可复现后，扩展 seed 池并开始批量采集高价值 run。

## Decision

可以进入下一阶段，但下一阶段应定义为“稳定自训练数据闭环 + 固定 seed / seed 池基础能力”，而不是直接开始大规模模型训练。
