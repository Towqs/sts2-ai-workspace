# Fixed Seed and Seed Pool Roadmap

更新日期：2026-05-12

## Summary

固定 seed 功能用于可复现的数据采集、问题诊断和模型对比，不作为默认长期训练模式。单个固定 seed 的 UI、Python runner、C# start_new_run 和数据记录链路已经落地；下一步是扩展为 seed 池，用于批量生产可追踪的高价值 run 数据。

当前这份文档保留固定 seed 的设计边界和 seed pool 后续计划。当前状态和训练路线以 `ai_training_roadmap.md` 为主。

## Product Intent

- 让自训练可以在指定 seed 上重复开局，便于复现同一地图、奖励、敌人和随机结果。
- 将通关或高进度 seed run 标记为高价值数据，和人工数据一起进入训练样本池。
- 避免长期重复训练少数 seed，防止模型记忆单局路线而不是学习通用策略。
- 将 seed 作为 run 数据的一部分保存，后续排查时可以知道每条数据来自哪个 seed。

## Current Status: Single Fixed Seed

- 控制面板自训练设置已支持 `self_play_seed`。
- 空 seed 保持随机开局行为。
- `control_state.json -> self_play_runner.py -> start_new_run` 已传递 seed。
- C# Mod `ExecuteStartNewRun` 已接收 seed，并调用角色选择/开局入口尝试应用 seed。
- `RL_DataCollector` 已记录 pending/current run seed，并把 seed 写入记录和 session。
- `self_play_scores.json` 已能显示 seed；最近 `seed=100` 的 self-play run 可用于复盘。

## Remaining Verification

- 同一 seed 连续开局时，首层地图、奖励和敌人是否完全一致，需要继续在真实游戏内验证。
- 当前单 seed 适合诊断死因，不适合长期重复训练。
- fixed seed run 的失败数据应进入复盘/偏好/惩罚数据池，不能因为可复现就默认进入 BC/RL 正样本。

## Seed Pool Extension

- 新增 seed 池配置：手动列表、文件导入或批量生成。
- 自训练每局从 seed 池取一个 seed，记录 seed、run_id、模型版本和结果。
- 支持训练集、验证集、测试集 seed 分离；测试 seed 永不入训。
- 对同一个 seed 的重复通关设置保留上限，避免数据重复污染。
- 对通关、Act 2+、F18+ 等高价值 run 做更高优先级入训。
- seed 池应优先覆盖不同路线、怪物组合、Boss 前后、关键稀有卡和已知失败点。

## Data Policy

- 通关 run 是高价值数据，但不自动要求保留当时产出的模型包。
- 数据和模型快照保持解耦：run 数据可以入训，模型是否保存仍由训练流程决定。
- 固定 seed 数据必须带上 seed、run_id、policy_name、model_version、质量标签和入训原因。
- 同 seed 的失败 run 也应保留评分，用于分析模型在哪些节点反复失败。

## Test Plan

- UI：输入 seed、保存、刷新后仍保留；清空后恢复随机。
- 配置链路：`control_state.json -> self_play_runner.py -> start_new_run` 中 seed 完全一致。
- 游戏端：同一 seed 连续开局，首层地图、奖励和敌人一致。
- 随机回归：空 seed 保持现有随机开局行为。
- 数据：run summary 和 `self_play_scores.json` 能显示该 run 使用的 seed。
- 回归：普通 AI 托管、手动开局、继续旧 run 不受固定 seed 设置影响。

## Updated Implementation Order

1. 继续验证单个 seed 在真实游戏内的可复现性。
2. 把 `seed=100` 这类稳定失败点纳入复盘，定位死因。
3. 实现 seed 池配置和轮换策略。
4. 支持训练/验证/测试 seed 分离。
5. 对 admitted seed run 接入 Phase 2A 微调；失败 seed run 进入复盘/偏好/惩罚数据池。
