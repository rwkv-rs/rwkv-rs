# 安装

## 前置条件

### Rust

需要 Rust 1.80 或更高版本(edition 2024). 如果你希望在 Windows 系统下配置, 建议访问 rustup.rs 下载并执行 rustup-init.exe .

```bash
# To install Rust, if you are running Unix/Linux/WSL
# run the following in your terminal, then follow the onscreen instructions.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 验证版本 / Verify version
rustc --version  # rustc 1.80.0 或更高
```

### 硬件后端

根据你的硬件选择对应的 feature flag：

Choose the feature flag matching your hardware:

| Feature | 硬件 / Hardware | 前置依赖 / Prerequisites |
|---|---|---|
| `cuda` | NVIDIA GPU | CUDA Toolkit 12.x, cuDNN |
| `rocm` | AMD GPU | ROCm 6.x |
| `metal` | Apple Silicon / AMD Mac | macOS 13+ |
| `vulkan` | 跨平台 GPU / Cross-platform | Vulkan SDK |
| `wgpu` | 通用 GPU / Universal | 无额外依赖 / None |

<div class="warning">

**CUDA 用户注意 / CUDA users note**

首次编译时，CubeCL 会为你的 GPU 编译 CUDA kernel，这可能需要 5-10 分钟。后续编译会使用缓存，速度正常。

</div>

---

## 获取源码 / Get the Source

```bash
git clone https://github.com/rwkv-rs/rwkv-rs
cd rwkv-rs
```

---

## 构建 / Build

```bash
# 构建推理示例（CUDA）/ Build inference example (CUDA)
cargo build -p rwkv-lm --example rwkv-lm-infer --release --features cuda

# 构建训练示例（CUDA）/ Build training example (CUDA)
cargo build -p rwkv-lm --example rwkv-lm-train --release --features cuda

# 使用 Metal（Apple Silicon）/ Using Metal (Apple Silicon)
cargo build -p rwkv-lm --example rwkv-lm-infer --release --features metal
```

<details>
<summary><strong>🦀 Rust Note：Cargo features 与 PyTorch 后端选择对比</strong></summary>

在 PyTorch 中，你通过安装不同的包来选择后端（`torch-cuda`, `torch-cpu`）。在 rwkv-rs 中，后端通过 Cargo feature flags 在编译时选择。这意味着：

- 不同后端的代码在编译时完全分离，没有运行时开销
- 同一份代码可以编译为不同后端的二进制
- 类似于 PyTorch 的 `torch.cuda.is_available()`，但在编译期就确定了

</details>

---

## 获取预训练权重 / Get Pretrained Weights

rwkv-rs 使用 `.mpk`（MessagePack）格式的权重文件，这是 Burn 框架的原生格式。

```bash
# 在 examples/rwkv-lm/ 目录下创建 weights 目录
mkdir -p examples/rwkv-lm/weights

# 从 HuggingFace 下载（需要先转换格式，参见 export 章节）
# Download from HuggingFace (requires format conversion, see export chapter)
```

<div class="warning">

目前需要将 HuggingFace 上的 RWKV-7 权重（`.pth` / `.safetensors`）转换为 `.mpk` 格式。转换工具参见 [导出章节](../export/overview.md)。

</div>
