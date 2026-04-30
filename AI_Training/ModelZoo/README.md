# ModelZoo

这里放可直接切换的推理模型包。每个模型包保持和运行目录一致的结构：

- `ProcessedParams/`：战斗 BC、候选动作评分模型、战斗 vocab 和 metadata。
- `ProcessedMacroParams/`：宏观 BC、宏观 vocab 和 metadata。
- `manifest.json`：模型包说明。

控制台的“AI 模型状态”里可以选择模型包并切换。切换会把模型包复制到当前运行目录，之后需要重启 AI 进程才会重新加载模型。
