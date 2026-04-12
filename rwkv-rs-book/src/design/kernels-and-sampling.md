# Kernel 与采样设计

`rwkv-rs` 的 kernel 与采样实现同时受到三类约束:

- 算法正确性: 需要回到论文公式与上游实现检查语义是否一致.
- 工程边界: 需要说明当前后端, device, dtype 与 shape 契约.
- 性能目标: 需要给出线程, 寄存器, 向量化访存与批调度的依据.

`rapid_sample` 是一个典型例子:

- 源码位置: `crates/rwkv-nn/src/kernels/rapid_sample/mod.rs`
- 对应微基准: `crates/rwkv-nn/benches/kernels/rapid_sample/forward.rs`

阅读这类模块时, 应优先找三类信息:

1. 术语段说明当前命名对应什么概念.
2. 函数注释说明 `Errors` 与 `Panics` 的边界.
3. 代码旁的设计说明与对应基准结果共同支撑论文依据与性能结论; 长期趋势以 CI benchmark 历史为准.
