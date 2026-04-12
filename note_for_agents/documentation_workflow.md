# 注释与文档流程

## 目标
- 让 public API 的 rustdoc 能持续补齐, 通过编译器 warning 暴露文档债务.
- 让复杂实现的设计意图, 关键约束与背景信息留在源码或 `rwkv-rs-book`, 便于长期维护.
- 避免引入偏离 Rust 社区常见做法的自定义文档门禁.

## 源码注释
- 优先使用标准 rustdoc 风格为 public API 编写文档.
- 对内部实现, 只在存在非显然约束, 算法背景或容易误用的路径时补充注释.
- 需要说明错误行为时, 优先使用 rustdoc 常见章节, 例如 `# Errors` 或 `# Panics`, 而不是自定义标签格式.
- `unwrap` / `expect` / `panic!` 在本仓库允许使用; 是否合适由具体语义和 code review 判断, 不做额外工具禁用.

## `rwkv-rs-book`
- `rwkv-rs-book` 用于整理用户可见能力, 设计背景, 历史演进与较长篇幅的实现说明.
- 当变更满足以下任一条件时, 应同步更新 `rwkv-rs-book`:
  - 调整了公共 API 或用户可见行为.
  - 修改了核心概念命名或设计解释.
  - 引入新的算法变体, 调度策略或性能结论.
  - 修正了过去设计文档中的假设.

## 审查与门禁
- 全仓 crate / example 根入口统一开启 `#![warn(missing_docs)]` 与 rustdoc 相关 `warn`, 用于暴露缺文档, 坏链接与无效 HTML 标签.
- CI 与日常验证主要依赖:
  - `cargo check --workspace`
  - `cargo doc --workspace --no-deps`
  - `mdbook build rwkv-rs-book`
- 注释内容的充分性, 错误信息是否清晰, 以及是否需要补设计背景, 主要通过 code review 保证.
