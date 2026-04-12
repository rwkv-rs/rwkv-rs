# 架构导览

`rwkv-rs` 不是单一模型实现, 而是一组围绕 RWKV 生态拆分的工程模块:

- `rwkv-nn`: 模型结构, kernel, 数学函数与 Burn 后端集成.
- `rwkv-train`: 训练, 优化器, 学习率调度与训练期观测.
- `rwkv-infer`: 推理服务, 采样, 排队调度, HTTP / IPC 暴露层.
- `rwkv-bench`: benchmark 与 profiling 支撑.
- `rwkv-eval`: benchmark 数据集与评估执行框架.

阅读顺序建议:

1. 先读 [核心术语](./terminology.md), 建立命名映射.
2. 再读 [异常策略](./error-strategy.md), 理解何时返回错误, 何时直接 `panic`.
3. 然后进入 [Kernel 与采样设计](./kernels-and-sampling.md) 和 [性能证据方法](./performance-evidence.md).

源码中的函数注释负责解释局部设计, 本书负责把跨模块关系串起来.
