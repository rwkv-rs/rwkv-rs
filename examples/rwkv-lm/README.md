# Language Model Pretrain

This project provides an example implementation for training next token prediction
models on minipile datasets using the Burn-based RWKV Deep Learning Library.

## Dataset Details
- [The MiniPile Challenge for Data-Efficient Language Models](https://arxiv.org/abs/2304.08442)
- MiniPile is a 6GB subset of the [deduplicated The Pile corpus](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated).
- More details on the MiniPile curation procedure and some pre-training results be found in the [MiniPile paper](https://arxiv.org/abs/2304.08442).
- For more details on the Pile corpus, we refer the reader to [the Pile datasheet](https://arxiv.org/abs/2201.07311).
- You can download the dataset from [BlinkDL/minipile-tokenized](https://huggingface.co/datasets/BlinkDL/minipile-tokenized)

# Usage

## CUDA backend (Recommended)

```bash
git clone https://github.com/rwkv-rs/rwkv-rs.git
cd rwkv-rs

# Use the --release flag to really speed up training.
# Add the f16 feature to run in f16. 
cargo run --example rwkv-lm-train --release --features cuda   # Train on the minipile dataset
```

## WGPU backend

```bash
git clone https://github.com/rwkv-rs/rwkv-rs.git
cd rwkv-rs

# Use the --release flag to really speed up training.
cargo run --example rwkv-lm-train --release --features wgpu   # Train on the minipile dataset
```

## Metal backend

```bash
git clone https://github.com/rwkv-rs/rwkv-rs.git
cd rwkv-rs

# Use the --release flag to really speed up training.
# Add the f16 feature to run in f16. 
cargo run --example rwkv-lm-train --release --features metal   # Train on the minipile dataset
```
