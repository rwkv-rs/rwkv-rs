# RWKV Distill Data Pipeline

这个 example 提供一条可断点续跑的蒸馏数据流水线：

1. 从输入 JSONL 读取单轮 `User/Assistant` 样本；
2. 调用 generator 模型，重写出多个用户问题变体；
3. 按 `task_id` 稳定分配 answer model，生成蒸馏回答；
4. 输出最终训练数据到 `train_jsonl_path`；
5. 输出可恢复状态到 `state_jsonl_path`。

## Input Schema

输入 JSONL 每行至少需要这些顶层字段：

- `task_id`
- `sample_index`
- `completions_id`
- `context`

其中 `context` 必须是单轮格式：

```text
User: ...
Assistant: ...
```

程序只会抽取其中的 `User:` 内容，作为重写问题的来源。

## Config

参考 [example.toml](/home/chase/GitHub/rwkv-rs/examples/rwkv-distill-data/config/example.toml)。

默认会输出两个文件：

- `output.state_jsonl_path`
  用于断点续跑的任务状态清单；
- `output.train_jsonl_path`
  最终蒸馏训练数据。

## Run

从仓库根目录运行：

```bash
cargo run -p rwkv-distill-data --example rwkv-distill-data -- synthesize
```

指定配置文件：

```bash
cargo run -p rwkv-distill-data --example rwkv-distill-data -- \
  synthesize \
  --config examples/rwkv-distill-data/config/example.toml
```

临时覆盖输入条数：

```bash
cargo run -p rwkv-distill-data --example rwkv-distill-data -- \
  synthesize \
  --config examples/rwkv-distill-data/config/example.toml \
  --limit 100
```
