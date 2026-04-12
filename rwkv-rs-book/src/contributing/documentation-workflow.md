# 注释与文档工作流

贡献者在提交实现前, 需要同时完成四件事:

1. 为受影响的 public API 补齐或更新 rustdoc.
2. 对复杂实现补充必要的设计说明, 不必强行给每个函数写样板注释.
3. 判断 `rwkv-rs-book` 是否需要同步更新.
4. 为性能或算法结论保留足够的上下文, 便于 review 和后续追溯.

仓库门禁入口:

```bash
cargo check --workspace
cargo doc --workspace --no-deps
mdbook build rwkv-rs-book
```

如果变更会影响用户理解某个核心概念, 就不能只更新源码注释, 必须同步更新本书对应章节.
