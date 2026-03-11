pub mod coding;
pub mod function_calling;
pub mod hf_downloader;
pub mod hf_viewer;
pub mod instruction_following;
pub mod knowledge;
pub mod maths;
pub mod parquet_utils;

use linkme::distributed_slice;
use once_cell::sync::Lazy;
use std::collections::BTreeMap;
use crate::inferers::{CompletionRequest, CompletionResponse};
use crate::runtime::OpenAiClient;


pub struct BenchmarkInfo {
    pub name: BenchmarkName,
    pub field: Field,
    pub display_name: &'static str,
    pub cot_mode: &'static [CoTMode],
    pub sampling_config: SamplingConfig,
    pub avg_ks: &'static [u8],
    pub pass_ks: &'static [u8],
    pub with_llm_judger: bool,
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


pub trait Benchmark: Send + Sync {
    type Item;

    fn load(&mut self);
    fn check(&self) -> bool;  // return need_download
    fn download(&self);

    fn get_expected_context(&self, item: &Self::Item, cot_mode: CoTMode) -> String;
    fn get_ref_answer(&self, item: &Self::Item) -> String;
    async fn answer_and_judge(
        &self,
        model_name: String,
        model_client: &OpenAiClient,
        judger_client: Option<&OpenAiClient>,
        cot_mode: CoTMode,
        item: &Self::Item,
    ) -> bool;
}


#[distributed_slice]
pub static ALL_BENCHMARKS: [BenchmarkInfo] = [..];

pub static BENCHMARKS_BY_FIELD: Lazy<BTreeMap<Field, Vec<&'static BenchmarkInfo>>> = Lazy::new(|| {
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


pub fn get_prompt_for_cot(expected_context: &String) -> String {
    expected_context
        .split_once("<|completions_of_cot|>")
        .unwrap()
        .0
        .to_string()
}


pub async fn get_completions_of_cot(
    model_client: &OpenAiClient,
    model_name: &String,
    prompt_for_cot: &String,
    sampling_config: &SamplingConfig,
) -> String {
    let req = CompletionRequest::new(
        model_name.clone(),
        prompt_for_cot.into(),
        vec!["</think>".to_string()],
        4096,
        &sampling_config,
        None,
        None,
    );

    let resp: CompletionResponse = model_client.completions()
        .create_byot(&req).await.unwrap();

    resp.choices[0].text.clone()
}


pub fn get_prompt_for_final_answer(
    expected_context: &String,
    completions_of_cot: Option<&String>,
) -> String {
    completions_of_cot
        .map(|cot| expected_context.replace("<|completions_of_cot|>", cot))
        .unwrap_or_else(|| expected_context.clone())
        .split_once("<|logprobs_of_choices|>")
        .unwrap()
        .0
        .to_string()
}