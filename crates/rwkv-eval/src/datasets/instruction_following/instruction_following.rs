use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, SamplingConfig,
};
use crate::evaluators::instruction_following::{
    InstructionSpec, build_prompt, describe_instructions, evaluate_response, generate_response,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};
use tokio::runtime::Runtime;

#[distributed_slice(ALL_BENCHMARKS)]
static IFEVAL_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("ifeval"),
    field: Field::InstructionFollowing,
    display_name: "IFEval",
    cot_mode: &[CoTMode::NoCoT],
    sampling_config: SamplingConfig {
        temperature: 0.3,
        top_k: 50,
        top_p: 0.3,
        presence_penalty: 0.5,
        repetition_penalty: 0.5,
        penalty_decay: 0.99,
    },
    n_shots: &[0],
    avg_ks: &[4.0],
    pass_ks: &[1],
    with_llm_judger: false,
    create: |dataset_root| Box::new(Ifeval::new(dataset_root)),
};

pub struct Ifeval {
    dataset_root: PathBuf,
    test: Vec<IfevalItem>,
}

pub struct IfevalItem {
    key: usize,
    prompt: String,
    instructions: Vec<InstructionSpec>,
}

#[derive(Deserialize)]
struct RawIfevalItem {
    key: usize,
    prompt: String,
    instruction_id_list: Vec<String>,
    kwargs: Vec<Map<String, Value>>,
}

impl Ifeval {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

#[async_trait]
impl Benchmark for Ifeval {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self.dataset_root.join("ifeval").join("input_data.jsonl");
        if !path.is_file() {
            return true;
        }

        self.test = read_jsonl_items::<RawIfevalItem, _>(&path)
            .into_iter()
            .map(|row| {
                assert_eq!(
                    row.instruction_id_list.len(),
                    row.kwargs.len(),
                    "ifeval row {} has mismatched instruction_id_list/kwargs lengths",
                    row.key
                );

                let instructions = row
                    .instruction_id_list
                    .into_iter()
                    .zip(row.kwargs)
                    .map(|(id, args)| InstructionSpec::new(id, args))
                    .collect();

                IfevalItem {
                    key: row.key,
                    prompt: row.prompt,
                    instructions,
                }
            })
            .collect();

        self.test.is_empty()
    }

    fn check(&self) -> bool {
        self.test.len() != 541
    }

    fn download(&self) {
        let runtime = Runtime::new().unwrap();
        let downloaded_path = runtime.block_on(download_url_files(
            &self.dataset_root,
            "ifeval",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("input_data.jsonl"),
                url: "https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl".to_string(),
            }],
            1,
        ));
        println!("ifeval dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        match cot_mode {
            CoTMode::NoCoT => {}
            _ => panic!("ifeval only supports CoTMode::NoCoT"),
        }
        assert_eq!(n_shot, 0, "ifeval only supports 0-shot");

        build_prompt(&self.test[index].prompt)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        let item = &self.test[index];
        format!(
            "key={}\n{}",
            item.key,
            describe_instructions(&item.instructions)
        )
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        _judger_model_name: Option<&str>,
        _judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> bool {
        let prompt = self.get_expected_context(index, cot_mode, n_shot);
        let response = generate_response(
            model_client,
            model_name,
            &prompt,
            &IFEVAL_INFO.sampling_config,
        )
        .await;
        if response.trim().is_empty() {
            return false;
        }

        evaluate_response(&self.test[index].instructions, &response).all_passed
    }
}
