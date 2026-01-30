# rwkv-rs 依赖与 Feature 管理设计原则

本文档定义了 `rwkv-rs` 项目的模块化结构和条件编译（feature flags）的使用规范。其核心设计哲学为 **“唯一入口规则” (The Single Entry Point Rule)**，旨在为用户提供最简洁的 API 和最直观的使用体验。

## 核心判断

**用户只依赖 `rwkv`。**

`rwkv` crate 是整个生态系统唯一的入口（Facade）。所有子功能（如训练、量化、数据处理）和底层依赖（如 `burn` 的组件）都必须通过 `rwkv` crate 的 feature flags 来激活，并通过 `rwkv` 的模块进行访问。用户绝不应该被要求手动添加 `rwkv-*` 或 `burn-*` 的子 crate 作为依赖。

---

## 三大组件（开发者视角）

这是一个对项目维护者可见的内部结构。用户不应关心此细节。

### 1. 配置 (The Config): `rwkv-config`

- **职责**: 作为整个生态系统的基石，定义所有共享的“数据契约”（纯数据结构）和“行为契约”（Traits）。它是依赖图的根，除了 `burn` 的核心库外，不应有任何 `rwkv-*` 依赖。
- **设计哲学：组合优于继承**:
    - `rwkv-config` 的存在是为了促成一个健壮的组合式架构。
    - 例如，`ModelConfig` 和 `LoraConfig` 都在 `rwkv-config` 中定义。`rwkv-nn` 中的模型实现会依赖 `ModelConfig`，而 `rwkv-train` 中的训练器则会同时依赖 `ModelConfig` 和 `LoraConfig` 来组合出完整的训练流程。
    - 这种方式将“数据”和“使用数据的逻辑”完全解耦，从根本上避免了循环依赖的可能。
- **内容**: `trait Model`, `struct ModelConfig`, `struct LoraConfig`, `struct TrainConfig` 等所有跨 crate 共享的定义。
- **规则**: `rwkv` (Facade) 负责将其内容重导出到 `rwkv::config` 或 `rwkv::model` 模块下。

### 2. 立面 (The Facade): `rwkv`

- **职责**: 作为用户的**唯一依赖项**和 API 入口。
- **规则**:
    - 其 `Cargo.toml` 是整个工作区的 **feature 总控制中心**。
    - 负责激活对应的 `rwkv-*` 和 `burn-*` 插件。
    - 负责将插件中的核心 API 重导出到逻辑一致的模块下（例如 `rwkv::train::...`, `rwkv::data::...`）。

### 3. 插件 (The Plugins): `rwkv-*`

- **职责**: 提供专业的、可选的功能。
- **定位**: `rwkv-infer`, `rwkv-train`, `rwkv-data`, `rwkv-quantize` 等。
- **规则**:
    - **总是**作为 `rwkv` crate 的可选依赖。
    - **绝不**应成为用户的直接依赖。

---

## Feature Flag 设计 (`rwkv/Cargo.toml`)

Feature flags 应按用户意图分类，而非实现细节。

### 1. 后端 (Backend Features)

- **作用**: 决定计算后端。
- **Features**: `wgpu` (默认), `cuda`, `rocm`, `vulkan`, `metal`, `webgpu`, `cpu` 等。
- **规则**: `rwkv` 负责将这些 feature 级联到所有需要的 `burn-*` 和 `rwkv-*` 依赖上。

### 2. 核心能力 (Capability Features)

- **作用**: 启用项目的主要功能模块。
- **Features**:
    - `infer`: (默认启用) 提供推理能力。
    - `train`: (可选) 提供训练所需的一切。**此 feature 必须同时激活 `burn-train`, `burn-data`, `rwkv-train` 等依赖，并将它们的 API 重导出到 `rwkv::train` 和 `rwkv::data` 模块中。**
    - `quantize`: (可选) 提供量化能力。
    - `agent`: (可选) 提供 Agent 相关能力。

### 3. 模型格式 (Format Features)

- **作用**: 控制模型加载格式的支持。
- **Features**:
    - `safetensors`: (默认启用) 支持 `safetensors` 格式。
    - `gguf`: (可选) 未来用于支持 GGUF 格式。

---

## 导入规则与实例 (用户视角)

### 规则总结

1.  **只依赖 `rwkv`**: 你的 `Cargo.toml` 里只有一个 `rwkv`。
2.  **用 Feature 声明意图**: 在 `rwkv` 的 `features` 列表里告诉它你想做什么（比如 `train`）。
3.  **所有 `use` 都从 `rwkv` 开始**: `use rwkv::custom::backend::...`, `use rwkv::train::...`, `use rwkv::model::...`。

### 示例 `Cargo.toml`

**场景1：一个简单的推理应用**
*你的意图：我需要用默认的 GPU 后端跑推理。*
```toml
# 用户的 Cargo.toml
[dependencies]
rwkv = "..."
```
*`rwkv` 的默认 features (`["infer", "wgpu", "safetensors"]`) 已为你准备好一切。*

**场景2：一个使用 `cpu` 后端进行训练的应用**
*你的意图：我需要训练，但我没有 GPU，所以用 CPU (`cpu`)。*
```toml
# 用户的 Cargo.toml
[dependencies]
rwkv = { 
    version = "...", 
    default-features = false, 
    features = ["train", "cpu"] 
}
```
*这就够了。你不需要知道 `burn-train` 的存在。*

### 示例代码

```rust
// 用户的 main.rs

// 从 rwkv::custom::backend 选择你的计算后端
use rwkv::custom::backend::{Wgpu, Cpu};

// 从 rwkv::model 获取模型定义
use rwkv::model::{Model, ModelConfig};

// 如果 "train" feature 开启，你可以访问训练和数据相关的工具
#[cfg(feature = "train")]
use {
    rwkv::train::{LearnerBuilder, TrainStep, ClassificationOutput},
    rwkv::data::{Dataset, DataLoaderBuilder, /* ... */}
};

fn main() {
    // 你的应用代码...
}
```

这个设计将所有复杂性都封装在了 `rwkv` crate 的内部，为最终用户提供了一个极其干净、直观且可预测的体验。这才是我们应该遵循的设计。
