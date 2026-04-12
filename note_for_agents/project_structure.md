# 项目结构与模块组织

## 仓库定位
本项目目标是构建围绕 RWKV 模型的完整 Rust 生态. 当前不少模块仍处于建设期, 部分描述代表规划方向, 不应默认视为现成实现.

## 目录概览
- `crates/`: 核心库代码.
- `examples/`: 端到端示例, 实验入口与集成验证脚本.
- `.github/workflows/`: 部署工作流.
- `target/`: 构建产物, 不作为源码修改.

## 主要 crate 职责
- `rwkv`: 统一导出入口与 feature 门面层.
- `rwkv-config`, `rwkv-derive`: 配置契约, 派生宏与配置加载.
- `rwkv-data`: 数据清洗流水线, mmap 数据集读写, 采样逻辑.
- `rwkv-nn`: 模型结构, 模块, 层, 函数与高性能算子实现.
- `rwkv-train`: 预训练, 后训练, PEFT, StateTuning 等训练能力.
- `rwkv-infer`: 高性能推理引擎, IPC/HTTP 接口与后端集成.
- `rwkv-eval`: 各类 LLM Benchmark 评估.
- `rwkv-export`: 模型权重映射与导出.
- `rwkv-prompt`: prompt 与 StateTuning 效果研究.
- `rwkv-trace`: 可解释性与训练/推理过程分析.
- `rwkv-agent`: 事件循环, 工具调用, 记忆与交互式 agent 能力.
- `rwkv-bench`: 性能基准测试与瓶颈分析.

## 示例与资源
- `examples/text-data-clean-pipeline`: 数据清洗流水线示例.
- `examples/paired-dna-mmap-dataset-converter`: mmap 数据转换示例.
- 未来若存在 `data/`, `weights/` 等大体量目录, 应视为资产, 权重, 避免将实现逻辑散落其中.

## 结构约束
所有子 crate 的依赖版本应由 workspace 统一管理. 新增依赖时, 优先在根 `Cargo.toml` 的 `workspace.dependencies` 中声明, 再在子 crate 中引用.
