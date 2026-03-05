# WIP 功能 / Work in Progress Features

以下 crate 正在积极开发中, 尚未达到稳定状态. 本章提供规划说明. 

---

## 🚧 rwkv-agent — Agent 框架 / Agent Framework

基于 RWKV 的 Agent 框架, 规划功能:

- **事件循环**: 异步事件驱动的 Agent 运行时
- **MCP 工具集成**: 支持 Model Context Protocol 工具调用
- **RLVR 训练**: 基于验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards）
- **记忆系统**: 长期记忆存储与检索, 利用 RWKV State 的高效性

Agent 框架将与 `rwkv-infer` 深度集成, 实现低延迟的本地推理调用.

---

## 🚧 rwkv-eval — Benchmark 评测 / Evaluation Suite

全面的 benchmark 评测套件, 规划功能:

- **nemo-skills 集成**: 支持 NVIDIA nemo-skills 全套 benchmark
- **标准 benchmark**: MMLU、HellaSwag、ARC、WinoGrande 等
- **多语言评测**: 中文、英文及其他语言的评测
- **自定义评测**: 支持用户定义的评测任务

---

## 🚧 rwkv-prompt — Prompt 优化 / Prompt Optimization

Prompt 工程与优化工具, 规划功能:

- **最优 Prompt 搜索**: 自动搜索特定任务的最优 prompt 格式
- **StateTuning 本质研究**: 研究 StateTuning 与 Prompt Tuning 的等价性
- **Prompt 压缩**: 将长 prompt 压缩为产生等价行为的短Prompt, 减少推理时的 prefill 开销

---

## 🚧 rwkv-trace — 激活值与梯度分析 / Activation & Gradient Analysis

模型可解释性工具, 规划功能: 
- **激活值操作**: 在推理时修改中间层激活值, 研究模型行为
- **权重操作**: 分析权重对模型输出的影响
- **梯度动力学**: 可视化训练过程中的梯度流动
- **State 可视化**: 可视化 WKV State 的演化过程, 理解模型的"记忆"机制

---

## 贡献 / Contributing

如果你对以上任何功能感兴趣, 欢迎在 [GitHub](https://github.com/rwkv-rs/rwkv-rs) 上提交 Issue 或 PR. 

If you're interested in any of the above features, feel free to open an Issue or PR on [GitHub](https://github.com/rwkv-rs/rwkv-rs).
