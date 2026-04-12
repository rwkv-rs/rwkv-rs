# 构建, 测试与开发流程

## 常用命令
- `cargo check --workspace`: 快速检查整个 workspace.
- `cargo build --workspace --release`: 构建 release 产物.
- `cargo +nightly fmt --all`: 按仓库 rustfmt 配置格式化代码.
- `cargo clippy --workspace --all-targets -- -D warnings`: 以零告警为目标检查代码.
- `cargo nextest run --workspace`: 优先使用的测试入口.
- `cargo bench -p <crate> --bench <name>`: 执行 Criterion 基准; 需要给 `github-action-benchmark` 留存时, 追加 `-- --output-format bencher`.
- `cargo doc --workspace --no-deps`: 生成文档并检查导出层; crate 根上的 `missing_docs` / rustdoc lint 会以 `warn` 暴露全仓文档问题.
- `mdbook build rwkv-rs-book`: 构建书籍并检查章节结构.

## 执行策略
优先运行与改动范围匹配的最小命令, 例如 `cargo test -p rwkv-infer` 或 `cargo check -p rwkv-eval`; 在跨 crate 或公共接口变更时, 再补充 workspace 级检查.

涉及“是否需要补单测/基准, 是否存在性能回退, 哪些命令才算验收完成”时, 以 `note_for_agents/verification_standards.md` 为准. 该文档定义了按风险分级的最低验证门槛与结果记录方式.

涉及测试或基准应放在哪个路径, 如何从源码反查测试/基准时, 以 `note_for_agents/testing_layout.md` 为准.

## 测试约定
- 正确性测试统一放在镜像 `src/` 的 `tests/` 目录树中.
- 不再新增实现文件内 `#[cfg(test)]` 模块或同目录 `tests.rs`.
- 涉及异步逻辑时使用 `#[tokio::test]`.
- `rwkv-eval` 的 benchmark 数据集测试位于 `crates/rwkv-eval/tests/cores/datasets/**`, 默认被忽略, 需使用 `cargo nextest -p rwkv-eval --test benchmark_datasets --run-ignored only ...` 执行.
- `examples/rwkv-lm-eval/scripts/unit_test.sh` 展示了一个需要 `HF_TOKEN` 的测试入口.

## 性能相关改动
涉及性能优化时, 不要只给出主观判断. 应结合基准测试或关键指标, 明确变更前后差异与瓶颈位置.

若修改命中了 `rwkv-nn` kernel, `rwkv-lm` 推理路径或其他热点代码, 应优先复用镜像 `src/` 的 `benches/` 基准路径; 若无现成 bench, 需要按镜像规则补充新的基准文件. 需要长期对比的微基准优先走 Criterion 与 `github-action-benchmark` 的标准接入方式, 不再在源码树中单独维护 benchmark 摘要目录.

当前 CI benchmark 历史只追踪 `rwkv-nn` 的 Criterion 微基准. `examples/rwkv-lm/benches/inferring.rs` 仍是带外部模型配置的手动压测入口, 不纳入该历史曲线.

涉及源码注释, 术语或异常策略变更时, 同时遵循 `note_for_agents/documentation_workflow.md`.
