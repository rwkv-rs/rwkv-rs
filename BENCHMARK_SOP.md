# Benchmark 添加 SOP

本文档整理当前 `rwkv-eval` 中新增 benchmark 的固定流程、设计判断点和代码风格约束。重点不是“怎么先跑通”，而是“怎么按现有项目风格正确接入，并且后续可维护”。

适用范围：

- `crates/rwkv-eval/src/datasets/**/*.rs`

说明：

- 本文的大部分内容是通用 SOP，适用于所有 field。
- 文中出现的 `knowledge` / `maths` 规则，是当前已经明确落地的 field-specific 约束。
- 如果你接下来新增的是其它 field 的 benchmark，优先遵守通用 SOP，再参考“最相近 field 的现有实现”落具体写法。

推荐直接对照的模板文件：

- 通用单文件 benchmark 参考：`crates/rwkv-eval/src/datasets/knowledge/mmlu.rs`
- 数学类 JSONL 参考：`crates/rwkv-eval/src/datasets/maths/gsm8k.rs`
- 数学类平坦 parquet 参考：`crates/rwkv-eval/src/datasets/maths/simpleqa.rs`
- 数学类嵌套 parquet 参考：`crates/rwkv-eval/src/datasets/maths/answer_judge.rs`
- 数学类 gated Hugging Face parquet 参考：`crates/rwkv-eval/src/datasets/maths/hle.rs`

## 1. 先判断你要加的 benchmark 属于哪个 field / task family

先决定 benchmark 应该落到哪个 field，因为这会直接决定 prompt 结构、判分方式和 `mod.rs` 可复用的基础函数。

- `Field::Knowledge`
    - 本质是单选题。
    - 参考实现：`knowledge/mmlu.rs`
    - 参考答案是选项字母。
    - `knowledge/mod.rs` 负责 few-shot 示例拼接、选项拼接、choice logprob 判分。
- `Field::Maths`
    - 本质是自由作答题。
    - 参考实现：`maths/gsm8k.rs`、`maths/simpleqa.rs`、`maths/answer_judge.rs`
    - 当前统一只支持 `CoTMode::CoT`。
    - `maths/mod.rs` 只负责 CoT prompt、最终答案生成、LLM judge。
- 其它 field
    - 先找同 field 下最接近的现有 benchmark。
    - 如果该 field 还没有成熟 benchmark，就让对应 `mod.rs` 只保留 field 级共享基础能力，把 benchmark-specific 逻辑继续下沉到各 benchmark 文件。

不要为了“抽象统一”把不同 field 再包出一层公共模板。项目当前的设计就是按 field 分开，分别复用各自 `mod.rs` 里的最小公共能力。

## 2. 新增前必须先想清楚的事情

在写代码前，先把下面几件事定死，否则后面一定返工。

### 2.1 数据源是什么

必须先确定：

- 数据来自 Hugging Face、GitHub raw、还是别的 URL。
- 原始文件格式是什么：`parquet`、`jsonl`、`csv`、仓库目录。
- 需要哪些 split。
- 本地缓存目录应该叫什么。

规则：

- 原始文件是什么，就按原始文件直接读取。
- 不要把 parquet 先转成 json/jsonl 再读。
- 不要为了“通用解析”引入中间值树，能直接按原始 schema 读就直接读。
- 本地目录落在 benchmark 自己的 slug 下，不要落在 HF repo 名下。

例子：

- `simpleqa` 下载到 `dataset_root/simpleqa/...`
- `hle` 下载到 `dataset_root/hle/...`
- `answer_judge` 下载到 `dataset_root/answer_judge/...`

### 2.2 benchmark 的题型和正确率判定方式

必须判断它属于哪种：

- 标准单选题：直接比较选项索引/字母。
- 自由作答但可规整成字符串答案：保存规整后的参考答案，再交给 LLM judge。
- 本身就是“判断答案对错”的数据：把题目和参考答案、学生答案拼成 judge task，再让被测模型输出 verdict。

这里不要偷懒。benchmark 的“任务定义”必须在 `expected_context` 里表达清楚，而不是事后靠脆弱的字符串提取修补。

### 2.3 采样参数从哪里来

优先复用现有组别，而不是每个 benchmark 都造一套新参数。

当前经验上主要有两组：

- 数学推理类：`temperature = 0.55, top_k = 66, top_p = 0.79, presence_penalty = 0.14, repetition_penalty = 0.01, penalty_decay = 0.997`
- 自由作答/判题类：`temperature = 0.3, top_k = 500, top_p = 0.4, presence_penalty = 0.5, repetition_penalty = 0.1, penalty_decay = 0.99`

只有当 benchmark 的生成形态明显不同，才新增或改动采样参数。否则优先贴近现有相似 benchmark。

### 2.4 是否需要 LLM judge

判断标准：

- 如果参考答案是离散选项，通常不需要 LLM judge。
- 如果模型输出是自由文本，通常需要 LLM judge。

数学类当前默认思路：

- 被测模型先生成 CoT 和最终答案。
- judge 使用单独的 chat completions 接口。
- judge 只返回 JSON schema 里的 `is_passed: bool`。
- judge 失败重试 3 次；3 次都失败直接 panic。
- 不做“失败返回 false”这种语义错误的降级。

## 3. 文件放置和模块边界

### 3.1 入口注册

新增 benchmark 时，至少要改两处：

- 在对应 `mod.rs` 里 `pub mod xxx;`
- 在 benchmark 文件里通过 `#[distributed_slice(ALL_BENCHMARKS)]` 注册 `BenchmarkInfo`

### 3.2 `mod.rs` 的职责边界

这是强约束。

通用原则：

- field 级 `mod.rs` 只放该 field 真正共享的基础能力。
- benchmark-specific 的加载、规整、编排、判分细节，继续下沉到 benchmark 文件本身。
- 不要因为新增了某个 benchmark，就顺手往全局或 field 层堆一个只服务单个 benchmark 的 helper / common。
- 不要把只会在单个 benchmark 或单个注册点使用的配置提成全局变量；直接内联到对应位置。
- prompt 拼接、stop tokens、max tokens、`SamplingConfig` 默认都应放在 benchmark 文件本身内联实现，方便独立修改。
- `mod.rs` 不应该承载某个 benchmark family 专属的 prompt 模板、脚本模板、后处理协议或解码流程。

### `knowledge/mod.rs`

允许保留：

- `Example`
- few-shot 示例拼接
- choices 拼接
- 选项答案映射
- CoT / NoCoT / FakeCoT 的基础 prompt
- choice logprob 判分

### `maths/mod.rs`

只保留：

- 数学题 `expected_context` 基础模板
- CoT prompt 提取
- 最终答案 prompt 提取
- 最终答案生成
- judge 基础函数

不要放进去：

- benchmark-specific 解析
- benchmark-specific 规整
- benchmark-specific few-shot 组装
- “common” / “helper” / `define_maths_benchmark` 一类新的抽象层

加载、规整、编排逻辑应该继续下沉到各个 `maths/*.rs`。

### 3.3 同系列共享逻辑放 `*_common.rs`

如果多个 benchmark 属于同一个系列，而且它们稳定共享一套 prompt 结构、脚本模板、判分协议或数据规整逻辑，可以新增同目录下的 `*_common.rs`。

例如：

- `gpqa_common.rs`
- `human_eval_common.rs`
- `mbpp_common.rs`
- `evalplus_common.rs`

规则：

- `*_common.rs` 只服务同一个 benchmark family，不跨 unrelated family 复用。
- benchmark 之间共享的是“同系列协议”，才适合放到 `*_common.rs`。
- `SamplingConfig` 仍然优先内联在各自 `BenchmarkInfo`，不要因为数值一样就提成 family/global 常量。
- 如果只是一个 benchmark 自己用的 prompt/script，就继续留在文件内，不要抽 `common`。
- 不要把多个不相干 family 的协议 helper 混进一个 field 级 `script_templates.rs` / `common.rs`。

## 4. benchmark 文件的标准结构

benchmark 文件尽量与同类 benchmark 保持对称。标准顺序如下：

1. `use`
2. `#[distributed_slice(ALL_BENCHMARKS)] static XXX_INFO`
3. benchmark struct
4. item struct
5. `impl BenchmarkNameStruct { pub fn new(...) -> Self }`
6. 少量 benchmark-local 解析函数
7. `impl Benchmark for ...`

### 4.1 struct 设计约束

不要加无意义封装。benchmark 内部 item struct 只保留这个 benchmark 真正需要的字段。

推荐形态：

```rust
pub struct XxxItem {
    question: String,
    answer: String,
    subject: String,
}