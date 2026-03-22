pub mod coding;
pub mod function_calling;
pub mod instruction_following;
pub mod knowledge;
pub mod maths;
pub mod utils;

use crate::inferers::generate_text_completion;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use once_cell::sync::Lazy;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub struct BenchmarkInfo {
    pub name: BenchmarkName,
    pub field: Field,
    pub display_name: &'static str,
    pub cot_mode: &'static [CoTMode],
    pub sampling_config: SamplingConfig,
    pub n_shots: &'static [u8],
    pub avg_ks: &'static [f32],
    pub pass_ks: &'static [u8],
    pub with_llm_judger: bool,
    pub create: BenchmarkFactory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BenchmarkName(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Field {
    Knowledge,
    Maths,
    Coding,
    InstructionFollowing,
    FunctionCalling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CoTMode {
    NoCoT,
    FakeCoT,
    CoT,
}

pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,

    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub penalty_decay: f32,
}

#[derive(Debug, Clone)]
pub struct Record {
    pub context: String,
    pub answer: String,
    pub ref_answer: String,
    pub is_passed: bool,
    pub fail_reason: String,
}

pub type BenchmarkFactory = fn(PathBuf) -> Box<dyn Benchmark>;

#[async_trait]
pub trait Benchmark: Send + Sync {
    fn load(&mut self) -> bool; // return is_invalid
    async fn check(&self) -> bool; // return is_invalid
    async fn download(&self);
    fn len(&self) -> usize;

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String;
    fn get_ref_answer(&self, index: usize) -> String;
    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        judger_model_name: Option<&str>,
        judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record;
}

#[distributed_slice]
pub static ALL_BENCHMARKS: [BenchmarkInfo] = [..];

pub static BENCHMARKS_BY_FIELD: Lazy<BTreeMap<Field, Vec<&'static BenchmarkInfo>>> =
    Lazy::new(|| {
        let mut map: BTreeMap<Field, Vec<&'static BenchmarkInfo>> = BTreeMap::new();

        for info in ALL_BENCHMARKS {
            map.entry(info.field).or_default().push(info);
        }

        // 两百个 benchmark 后，顺序不要赌链接器/注册顺序，统一显式排序
        for vec_info in map.values_mut() {
            vec_info.sort_unstable_by_key(|m| m.name.0);
        }

        map
    });

pub fn get_benchmarks_with_field(field: Field) -> &'static [&'static BenchmarkInfo] {
    static EMPTY: &[&BenchmarkInfo] = &[];
    BENCHMARKS_BY_FIELD
        .get(&field)
        .map(Vec::as_slice)
        .unwrap_or(EMPTY)
}

pub fn apply_user_assistant_template(user_part: String, assistant_part: String) -> String {
    format!("User: {user_part}\n\nAssistant: {assistant_part}")
}

pub fn get_prompt_for_cot(expected_context: &str) -> String {
    expected_context
        .split_once("<|completions_of_cot|>")
        .unwrap()
        .0
        .to_string()
}

pub async fn get_completions_of_cot(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt_for_cot: &str,
    sampling_config: &SamplingConfig,
) -> String {
    generate_text_completion(
        model_client,
        model_name,
        prompt_for_cot,
        vec!["</think>".to_string()],
        4096,
        &sampling_config,
    )
    .await
    .unwrap()
}

pub fn get_prompt_for_final_answer(
    expected_context: &str,
    completions_of_cot: Option<&str>,
) -> String {
    completions_of_cot
        .map(|cot| expected_context.replace("<|completions_of_cot|>", cot))
        .unwrap_or_else(|| expected_context.to_string())
        .split_once("<|logprobs_of_choices|>")
        .unwrap()
        .0
        .to_string()
}

pub fn render_context(expected_context: &str, replacements: &[(&str, &str)]) -> String {
    let mut context = expected_context.to_string();
    for (placeholder, value) in replacements {
        context = context.replace(placeholder, value);
    }
    context
}

#[cfg(test)]
pub(crate) async fn assert_benchmark_download_load_and_read(info: &'static BenchmarkInfo) {
    let tempdir = tempfile::tempdir()
        .unwrap_or_else(|err| panic!("create tempdir failed for `{}`: {err}", info.name.0));
    let mut benchmark = (info.create)(tempdir.path().to_path_buf());

    benchmark.download().await;

    assert!(
        !benchmark.load(),
        "load() returned invalid after download for `{}`",
        info.name.0
    );
    assert!(
        !benchmark.check().await,
        "check() returned invalid after download for `{}`",
        info.name.0
    );
    assert!(benchmark.len() > 0, "len() == 0 for `{}`", info.name.0);

    let cot_mode = *info
        .cot_mode
        .first()
        .unwrap_or_else(|| panic!("benchmark `{}` has empty cot_mode", info.name.0));
    let n_shot = *info
        .n_shots
        .first()
        .unwrap_or_else(|| panic!("benchmark `{}` has empty n_shots", info.name.0));

    let expected_context = benchmark.get_expected_context(0, cot_mode, n_shot);
    assert!(
        !expected_context.trim().is_empty(),
        "get_expected_context(0, ..) returned empty text for `{}`",
        info.name.0
    );

    let ref_answer = benchmark.get_ref_answer(0);
    assert!(
        !ref_answer.trim().is_empty(),
        "get_ref_answer(0) returned empty text for `{}`",
        info.name.0
    );
}
