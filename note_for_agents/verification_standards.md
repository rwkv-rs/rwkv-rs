# 验证标准

## 基础要求
- 任何影响源码注释契约, 模块术语, 异常策略或证据引用的改动, 至少执行:
  - `cargo doc --workspace --no-deps`
  - `mdbook build rwkv-rs-book`
- 任何代码改动仍需执行与范围匹配的 `cargo check`, `cargo test` 或 `cargo nextest`.
- 测试与基准的放置路径必须符合 `note_for_agents/testing_layout.md`, 不再接受 `src/**` 内联测试.
- 全仓 crate / example 根默认用 `#![warn(missing_docs)]` 与 rustdoc `warn` 暴露文档债务.

## 风险分级
- 低风险:
  - 仅文档或注释调整, 无行为变化.
  - 最低要求: `cargo doc --workspace --no-deps`, `mdbook build rwkv-rs-book`.
- 中风险:
  - 修改公共 API, 数据结构, 配置加载或错误传播.
  - 最低要求: 低风险命令 + 相关 crate `cargo check` / `cargo test`.
- 高风险:
  - 修改 kernel, 推理热点, 调度逻辑, benchmark harness 或训练优化器.
  - 最低要求: 中风险命令 + 对应 benchmark 运行与结果确认.

## 结果留存
- 当变更涉及性能结论时, 优先更新对应 Criterion benchmark, 并以 `github-action-benchmark` 的 CI 历史页面作为长期留存.
- 当变更涉及概念或历史解释时, 应同步更新 `rwkv-rs-book`.
