# Fixed Seed and Seed Pool Roadmap

更新日期：2026-05-04

## Summary

固定 seed 功能用于可复现的数据采集、问题诊断和模型对比，不作为默认长期训练模式。第一阶段先支持单个固定 seed；确认闭环稳定后，再扩展为 seed 池，用于批量生产可追踪的高价值 run 数据。

当前计划只记录未来实现方案，不在本次提交中实现固定 seed。

## Product Intent

- 让自训练可以在指定 seed 上重复开局，便于复现同一地图、奖励、敌人和随机结果。
- 将通关或高进度 seed run 标记为高价值数据，和人工数据一起进入训练样本池。
- 避免长期重复训练少数 seed，防止模型记忆单局路线而不是学习通用策略。
- 将 seed 作为 run 数据的一部分保存，后续排查时可以知道每条数据来自哪个 seed。

## MVP: Single Fixed Seed

- 在控制面板自训练设置中新增 `self_play_seed`。
- 空 seed 保持现有随机开局行为。
- `control_state.json` 保存 seed，`self_play_runner.py` 读取并传给游戏 API。
- `start_new_run` 请求中的 seed 必须从控制面板一路传到 C# Mod。
- 每条 self-play run 的 `game_start`、macro 记录和 `self_play_scores.json` 都记录 seed。
- UI 显示当前 seed，避免误以为随机训练正在使用固定局。

## Required Game API Work

当前 C# Mod 仍然拒绝非空 seed：

- `训练脚本/STS2MCP/McpMod.Actions.cs` 中 `ExecuteStartNewRun` 会返回 `start_new_run currently supports seed=null only`。
- 当前 `RL_DataCollector.MarkMenuRunIntent("new", null)` 会丢弃 seed。
- 当前菜单启动辅助逻辑更偏向点按钮或调用无参方法；需要确认游戏实际开局入口能接收 seed。

未来实现必须补齐这一层，否则 Python 和 UI 即使保存了 seed，游戏也不会真正使用它。

## Seed Pool Extension

- 新增 seed 池配置：手动列表、文件导入或批量生成。
- 自训练每局从 seed 池取一个 seed，记录 seed、run_id、模型版本和结果。
- 支持训练集、验证集、测试集 seed 分离；测试 seed 永不入训。
- 对同一个 seed 的重复通关设置保留上限，避免数据重复污染。
- 对通关、Act 2+、F18+ 等高价值 run 做更高优先级入训。

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

## Implementation Order

1. 修稳当前 self-play 数据完整性和结束状态判定。
2. 做单个 `self_play_seed` 的 UI、配置和 runner 链路。
3. 修改 C# `start_new_run`，让游戏端真正接收 seed。
4. 将 seed 写入 run 数据和 self-play 评分。
5. 验证同 seed 可复现后，再实现 seed 池。
