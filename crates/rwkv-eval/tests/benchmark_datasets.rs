macro_rules! benchmark_dataset_tests {
    ($benchmark_name:literal) => {
        #[tokio::test]
        #[ignore = "downloads remote benchmark dataset into examples/rwkv-lm-eval/datasets"]
        async fn download_dataset() {
            crate::cores::datasets::assert_download_dataset($benchmark_name).await;
        }

        #[tokio::test]
        #[ignore = "loads benchmark dataset from examples/rwkv-lm-eval/datasets"]
        async fn load_dataset() {
            crate::cores::datasets::assert_load_dataset($benchmark_name).await;
        }

        #[tokio::test]
        #[ignore = "renders benchmark context from examples/rwkv-lm-eval/datasets"]
        async fn show_expected_context() {
            crate::cores::datasets::assert_show_expected_context($benchmark_name).await;
        }
    };
}

mod cores;
