export HF_TOKEN="hf_xxxxxxxxxx"
cargo nextest run -p rwkv-eval --lib --run-ignored only -E 'test(/(download_dataset|load_dataset|show_expected_context)$/)' --no-capture