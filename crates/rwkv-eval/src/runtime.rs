use std::collections::HashMap;
use std::path::PathBuf;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        chat::{
            ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
            CreateChatCompletionRequestArgs,
        },
        completions::CreateCompletionRequestArgs,
    },
};
use rwkv_config::{
    raw::eval::{ApiConfig, ApiConfig},
    validated::eval::FinalEvalConfig,
};

use crate::error::EvalError;

pub type OpenAiClient = Client<OpenAIConfig>;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelSelector {
    pub model_arch_version: String,
    pub model_data_version: String,
    pub model_num_params: String,
}

impl ModelSelector {
    pub fn new(model_arch_version: &str, model_data_version: &str, model_num_params: &str) -> Self {
        Self {
            model_arch_version: model_arch_version.to_string(),
            model_data_version: model_data_version.to_string(),
            model_num_params: model_num_params.to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedModelTarget {
    pub selector: ModelSelector,
    pub config: ApiConfig,
}

impl ResolvedModelTarget {
    pub fn display_name(&self) -> String {
        format!(
            "{}-{}-{} ({})",
            self.selector.model_arch_version,
            self.selector.model_data_version,
            self.selector.model_num_params,
            self.config.model
        )
    }
}

#[derive(Clone)]
pub struct ModelRuntime {
    pub target: ResolvedModelTarget,
    pub client: OpenAiClient,
}

#[derive(Clone)]
pub struct EvalRuntime {
    datasets_path: PathBuf,
    models: Vec<ModelRuntime>,
    judger: ServiceRuntime,
    checker: ServiceRuntime,
}

#[derive(Clone)]
struct ServiceRuntime {
    config: ApiConfig,
    client: OpenAiClient,
}

impl EvalRuntime {
    pub async fn new(cfg: &FinalEvalConfig, datasets_path: PathBuf) -> Result<Self, EvalError> {
        let mut model_services = HashMap::new();
        for model in cfg.models.iter() {
            model_services.insert(
                ModelSelector::new(
                    &model.model_arch_version,
                    &model.model_data_version,
                    &model.model_num_params,
                ),
                model.clone(),
            );
        }

        let mut models = Vec::new();
        for model_arch_version in &cfg.model_arch_versions {
            for model_data_version in &cfg.model_data_versions {
                for model_num_params in &cfg.model_num_params {
                    let selector = ModelSelector::new(
                        model_arch_version,
                        model_data_version,
                        model_num_params,
                    );

                    let config = model_services.get(&selector).cloned().ok_or_else(|| {
                        EvalError::Config(format!(
                            "missing model config for {}-{}-{}",
                            model_arch_version, model_data_version, model_num_params
                        ))
                    })?;

                    models.push(ModelRuntime {
                        target: ResolvedModelTarget {
                            selector,
                            config: config.clone(),
                        },
                        client: build_client(&config.base_url, &config.api_key),
                    });
                }
            }
        }

        let judger = ServiceRuntime {
            client: build_client(&cfg.llm_judger.base_url, &cfg.llm_judger.api_key),
            config: cfg.llm_judger.clone(),
        };
        let checker = ServiceRuntime {
            client: build_client(&cfg.llm_checker.base_url, &cfg.llm_checker.api_key),
            config: cfg.llm_checker.clone(),
        };

        let runtime = Self {
            datasets_path,
            models,
            judger,
            checker,
        };

        runtime.smoke_test().await?;
        Ok(runtime)
    }

    pub fn datasets_path(&self) -> &PathBuf {
        &self.datasets_path
    }

    pub fn models(&self) -> &[ModelRuntime] {
        &self.models
    }

    pub async fn complete_with_model(
        &self,
        model: &ModelRuntime,
        prompt: &str,
    ) -> Result<String, EvalError> {
        let request = CreateCompletionRequestArgs::default()
            .model(model.target.config.model.clone())
            .prompt(prompt)
            .max_tokens(128u32)
            .temperature(0.0)
            .build()?;

        let response = model.client.completions().create(request).await?;
        response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.text)
            .ok_or_else(|| EvalError::InvalidResponse("empty completion choices".to_string()))
    }

    pub async fn judge_answer(
        &self,
        benchmark_name: &str,
        context: &str,
        ref_answer: &str,
        final_answer: &str,
    ) -> Result<bool, EvalError> {
        let system_prompt =
            "You are a strict benchmark answer judge. Reply with PASS or FAIL only.";
        let user_prompt = format!(
            "Benchmark: {benchmark_name}\nExpected context:\n{context}\n\nReference answer: {ref_answer}\nModel final answer: {final_answer}\n\nDoes the model final answer match the reference answer?"
        );

        let response = self
            .chat_completion(&self.judger, system_prompt, &user_prompt)
            .await?;

        Ok(response.trim().to_ascii_uppercase().starts_with("PASS"))
    }

    pub async fn classify_error(
        &self,
        benchmark_name: &str,
        context: &str,
        ref_answer: &str,
        final_answer: &str,
    ) -> Result<String, EvalError> {
        let system_prompt = concat!(
            "You classify benchmark failures.",
            " Return exactly one label from: knowledge_gap, option_misread, format_error, reasoning_error, other."
        );
        let user_prompt = format!(
            "Benchmark: {benchmark_name}\nContext:\n{context}\n\nReference answer: {ref_answer}\nModel final answer: {final_answer}"
        );

        let response = self
            .chat_completion(&self.checker, system_prompt, &user_prompt)
            .await?;

        let label = response.trim().to_ascii_lowercase();
        let canonical = match label.as_str() {
            "knowledge_gap" | "option_misread" | "format_error" | "reasoning_error" | "other" => {
                label
            }
            _ => "other".to_string(),
        };

        Ok(canonical)
    }

    async fn smoke_test(&self) -> Result<(), EvalError> {
        for model in &self.models {
            let request = CreateCompletionRequestArgs::default()
                .model(model.target.config.model.clone())
                .prompt("Reply with OK.")
                .max_tokens(8u32)
                .temperature(0.0)
                .build()?;
            model.client.completions().create(request).await?;
        }

        self.smoke_test_chat(&self.judger).await?;
        self.smoke_test_chat(&self.checker).await?;

        Ok(())
    }

    async fn smoke_test_chat(&self, service: &ServiceRuntime) -> Result<(), EvalError> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(service.config.model.clone())
            .max_tokens(8u32)
            .temperature(0.0)
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content("Reply with OK.")
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content("OK")
                    .build()?
                    .into(),
            ])
            .build()?;

        service.client.chat().create(request).await?;
        Ok(())
    }

    async fn chat_completion(
        &self,
        service: &ServiceRuntime,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, EvalError> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(service.config.model.clone())
            .max_tokens(256u32)
            .temperature(0.0)
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt)
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content(user_prompt)
                    .build()?
                    .into(),
            ])
            .build()?;

        let response = service.client.chat().create(request).await?;
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| {
                EvalError::InvalidResponse("empty chat completion choices".to_string())
            })?;

        Ok(content)
    }
}

fn build_client(base_url: &str, api_key: &str) -> OpenAiClient {
    let config = OpenAIConfig::new()
        .with_api_key(api_key)
        .with_api_base(normalize_api_base(base_url));

    Client::with_config(config)
}

fn normalize_api_base(base_url: &str) -> String {
    let mut value = base_url.trim().trim_end_matches('/').to_string();
    if !value.contains("://") {
        value = format!("http://{value}");
    }
    if !value.ends_with("/v1") {
        value.push_str("/v1");
    }
    value
}

#[cfg(test)]
mod tests {
    use super::normalize_api_base;

    #[test]
    fn normalize_api_base_adds_scheme_and_v1() {
        assert_eq!(
            normalize_api_base("10.0.0.1:8080"),
            "http://10.0.0.1:8080/v1"
        );
        assert_eq!(
            normalize_api_base("https://demo.example.com/v1/"),
            "https://demo.example.com/v1"
        );
    }
}
