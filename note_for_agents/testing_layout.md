# 测试与基准目录规范

## 核心规则
- 所有正式测试都从源码目录剥离, 不允许继续写在 `src/**` 里的 `#[cfg(test)]` 模块.
- 正确性测试必须镜像 `src/` 目录树放在 `tests/`.
- 性能基准必须镜像 `src/` 目录树放在 `benches/`.
- 参考数据放在 `testdata/`, 参考数据生成脚本放在 `testgen/`.

## 路径映射
- 源码: `crates/<crate>/src/foo/bar.rs`
- 测试: `crates/<crate>/tests/foo/bar.rs`
- 基准: `crates/<crate>/benches/foo/bar.rs`

示例:
- `crates/rwkv-eval/src/cores/datasets/maths/gsm8k.rs`
- `crates/rwkv-eval/tests/cores/datasets/maths/gsm8k.rs`

- `crates/rwkv-nn/src/kernels/rapid_sample/forward.rs`
- `crates/rwkv-nn/benches/kernels/rapid_sample/forward.rs`

- `examples/rwkv-lm/src/inferring.rs`
- `examples/rwkv-lm/benches/inferring.rs`

## 模块组织
- `tests/` 和 `benches/` 下的目录结构应与 `src/` 对齐.
- 共享辅助代码统一放在镜像树对应层级的 `mod.rs`.
- 不再使用 `tests.rs`, `common.rs`, `helpers.rs`, `support.rs` 这类横向命名来承载正式测试入口.
- 若某个基准或测试覆盖多个入口, 应按主入口拆分, 不保留大而全的聚合文件.

## 命名约定
- 测试函数命名: `<被测函数>__<场景>__<结果>`
- 基准函数命名: `<被测函数>__<场景>`
- 禁止使用 `test_*`, `works`, `basic`, `happy_path`, `should_work` 这类弱语义名字.

## 执行入口
- `rwkv-eval` 的 benchmark 数据集测试统一从 `cargo nextest run -p rwkv-eval --test benchmark_datasets ...` 进入.
- 单个 crate 的 Criterion 基准通过 `cargo bench -p <crate> --bench <name>` 执行; 需要接入 CI benchmark 历史时, 额外追加 `-- --output-format bencher`.

## 约束
- Python 不参与正式测试编排, 只允许在 `testgen/` 中生成 `testdata/`.
- 改动源码时, 应同步维护对应镜像测试或镜像基准路径, 不允许出现只能靠经验查找的测试位置.
