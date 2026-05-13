# Project Progress: 2026-05-04

这是一份历史快照，保留用于追踪项目从“能否跑起来”进入“稳定自训练数据闭环”的节点。

当前状态已经更新：

- 单个 fixed seed 链路已落地到 UI、Python runner、C# start_new_run 和数据记录。
- 自训练能启动、采集、评分，并记录入训/拒绝原因。
- Phase 2A 已接入训练链路：`train_rl_finetune.py` 会使用 admitted self-play run 的 reward 权重微调候选动作模型；没有合格 run 时安全跳过。
- 最新路线、当前风险和下一步计划以 [`ai_training_roadmap.md`](ai_training_roadmap.md) 为准。
- 对外简版路线以 [`public_roadmap.md`](public_roadmap.md) 为准。

历史结论仍然有效：不要盲目扩大普通自训练规模。下一阶段应该围绕高价值数据、fixed seed 复盘、seed 池、评测闭环和 admitted self-play run 展开。
