use async_openai::{Client, config::OpenAIConfig};
use rwkv_config::raw::eval::IntApiConfig;
use rwkv_eval::{
    datasets::SamplingConfig,
    inferers::{CompletionRequest, CompletionResponse},
};

pub(crate) struct ClientWithConfig {
    pub api_cfg: IntApiConfig,
    pub client: Client<OpenAIConfig>,
}

pub(crate) fn build_client(base_url: &str, api_key: &str) -> Client<OpenAIConfig> {
    let config = OpenAIConfig::new()
        .with_api_key(api_key.to_string())
        .with_api_base(norm_api_url(base_url));

    Client::with_config(config)
}

fn norm_api_url(base_url: &str) -> String {
    let base_url = base_url.trim();
    assert!(!base_url.is_empty(), "base_url cannot be empty");

    let base_url = if base_url.contains("://") {
        base_url.to_string()
    } else {
        format!("http://{base_url}")
    };
    let base_url = base_url.trim_end_matches('/').to_string();

    if base_url.ends_with("/v1") {
        base_url
    } else {
        format!("{base_url}/v1")
    }
}

pub(crate) async fn check_client(client: &Client<OpenAIConfig>, model_name: &str) {
    let req = CompletionRequest::new(
        model_name.to_string(),
        "ping".into(),
        vec!["\n".to_string()],
        1,
        &SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            presence_penalty: 0.0,
            repetition_penalty: 0.0,
            penalty_decay: 1.0,
        },
        None,
        None,
    );

    let _: CompletionResponse = client
        .completions()
        .create_byot(&req)
        .await
        .unwrap_or_else(|error| panic!("client `{model_name}` is unavailable: {error}"));
}
