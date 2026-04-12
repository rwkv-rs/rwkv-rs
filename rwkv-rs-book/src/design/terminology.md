# 核心术语

源码中的术语源头不放在单独的中央总表, 而是放在各模块的 `mod.rs` / `lib.rs` 中.

这样做的理由:

- 术语与模块边界绑定, 避免“同名词在不同上下文下含义不同”却被强行合并.
- 读源码时可以直接在入口模块看到概念翻译, 不需要跳转到仓库另一处总表.
- review 时可以直接检查术语是否和模块实现一起更新.

当前建议关注的术语类别:

- 模型结构术语: `time_mixer`, `channel_mixer`, `state_adapter`.
- kernel 术语: `tile`, `line`, `batch_ids`, `penalty state`.
- 推理调度术语: `queue`, `schedule`, `forward`, `guided decode`.
- 训练术语: `grad clipper`, `lr scheduler`, `optimizer grouping`.

当源码命名与论文或上游仓库不完全一致时, 应在模块术语段说明:

- 上游原名
- 当前代码名
- 保留或改名原因
