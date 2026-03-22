use crate::datasets::maths::{get_final_answer_with_cot_mode, judge_with_retry};
use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::hf::viewer::get_parquet_files;
use crate::datasets::utils::parquet::{get_string, read_parquet_items};
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record,
    SamplingConfig, apply_user_assistant_template,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use parquet::record::Row;
use std::path::{Path, PathBuf};

const POLYMATH_DATASET: &str = "Qwen/PolyMath";
const POLYMATH_EXPECTED_LEN: usize = 18 * 4 * 125;
const POLYMATH_LANGUAGES: &[&str] = &[
    "ar", "bn", "de", "en", "es", "fr", "id", "it", "ja", "ko", "ms", "pt", "ru", "sw", "te", "th",
    "vi", "zh",
];
const POLYMATH_SPLITS: &[&str] = &["top", "high", "medium", "low"];

#[distributed_slice(ALL_BENCHMARKS)]
static POLYMATH_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("polymath"),
    field: Field::Maths,
    display_name: "PolyMath",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.55,
        top_k: 66,
        top_p: 0.79,
        presence_penalty: 0.14,
        repetition_penalty: 0.01,
        penalty_decay: 0.997,
    },
    n_shots: &[0],
    avg_ks: &[4.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(PolyMath::new(dataset_root)),
};

pub struct PolyMath {
    dataset_root: PathBuf,
    test: Vec<PolyMathItem>,
}

pub struct PolyMathItem {
    id: String,
    question: String,
    answer: String,
    language: String,
    difficulty: String,
}

impl PolyMath {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn get_answer_note(language: &str) -> &'static str {
    match language {
        "en" => r"Put the final answer in \boxed{}.",
        "zh" => r"请将最终答案放在 \boxed{} 中。",
        "ar" => r"ضع الإجابة النهائية داخل \boxed{}.",
        "bn" => r"চূড়ান্ত উত্তরটি \boxed{} এর মধ্যে দিন।",
        "de" => r"Setzen Sie die endgültige Antwort in \boxed{}.",
        "es" => r"Coloque la respuesta final en \boxed{}.",
        "fr" => r"Veuillez mettre la réponse finale dans \boxed{}.",
        "id" => r"Letakkan jawaban akhir di dalam \boxed{}.",
        "it" => r"Metti la risposta finale in \boxed{}.",
        "ja" => r"最終的な答えを \boxed{} に入れてください。",
        "ko" => r"최종 답안을 \boxed{} 안에 넣어 주세요.",
        "ms" => r"Letakkan jawapan akhir dalam \boxed{}.",
        "pt" => r"Coloque a resposta final em \boxed{}.",
        "ru" => r"Поместите окончательный ответ в \boxed{}.",
        "sw" => r"Weka jibu la mwisho katika \boxed{}.",
        "te" => r"తుది జవాబును \boxed{} లో ఉంచండి.",
        "th" => r"ใส่คำตอบสุดท้ายใน \boxed{}.",
        "vi" => r"Đặt câu trả lời cuối cùng trong \boxed{}.",
        _ => panic!("unsupported PolyMath language `{language}`"),
    }
}

fn get_language_control(language: &str) -> &'static str {
    match language {
        "en" => "Use English to think and answer.",
        "zh" => "使用中文进行思考和回答。",
        "ar" => "استخدم العربية في التفكير والإجابة.",
        "bn" => "চিন্তা ও উত্তরের জন্য বাংলা ব্যবহার করুন।",
        "de" => "Verwenden Sie Deutsch zum Denken und Antworten.",
        "es" => "Use español para pensar y responder.",
        "fr" => "Utilisez le français pour réfléchir et répondre.",
        "id" => "Gunakan bahasa Indonesia untuk berpikir dan menjawab.",
        "it" => "Usa l'italiano per pensare e rispondere.",
        "ja" => "日本語で考え、日本語で回答してください。",
        "ko" => "한국어로 생각하고 한국어로 답하세요.",
        "ms" => "Gunakan bahasa Melayu untuk berfikir dan menjawab.",
        "pt" => "Use português para pensar e responder.",
        "ru" => "Используйте русский язык для размышления и ответа.",
        "sw" => "Tumia Kiswahili kufikiri na kujibu.",
        "te" => "ఆలోచించడానికి మరియు సమాధానం ఇవ్వడానికి తెలుగు ఉపయోగించండి.",
        "th" => "ใช้ภาษาไทยในการคิดและตอบคำถาม",
        "vi" => "Sử dụng tiếng Việt để suy nghĩ và trả lời.",
        _ => panic!("unsupported PolyMath language `{language}`"),
    }
}

fn extract_boxed_answer(solution: &str) -> Option<String> {
    let start = solution.rfind(r"\boxed{")? + r"\boxed{".len();
    let tail = &solution[start..];
    let end = tail.find('}')?;
    Some(tail[..end].trim().to_string())
}

fn build_expected_context(item: &PolyMathItem, cot_mode: CoTMode) -> String {
    if cot_mode != CoTMode::CoT {
        panic!("polymath only supports CoT mode, got {cot_mode:?}");
    }

    let user_part = format!(
        "{}\n\n{}\n\n{}",
        item.question.trim(),
        get_language_control(&item.language),
        get_answer_note(&item.language),
    );
    let assistant_part = concat!(
        "<think><|completions_of_cot|></think>\n",
        "Therefore, the answer is \\(\\boxed{<|final_answer|>}\\)."
    )
    .to_string();

    apply_user_assistant_template(user_part, assistant_part)
}

fn parse_language_and_difficulty(path: &Path) -> (String, String) {
    let difficulty = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .unwrap_or_else(|| panic!("missing PolyMath split dir for {}", path.display()))
        .to_string();
    let language = path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        .unwrap_or_else(|| panic!("missing PolyMath language dir for {}", path.display()))
        .to_string();

    (language, difficulty)
}

#[async_trait]
impl Benchmark for PolyMath {
    fn load(&mut self) -> bool {
        self.test.clear();

        let parquet_paths =
            collect_files_with_extension(self.dataset_root.join("polymath"), "parquet");
        if parquet_paths.is_empty() {
            return true;
        }

        for path in parquet_paths {
            let (language, difficulty) = parse_language_and_difficulty(&path);
            let parse_item = |row: &Row| PolyMathItem {
                id: get_string(row, "id"),
                question: get_string(row, "question"),
                answer: get_string(row, "answer"),
                language: language.clone(),
                difficulty: difficulty.clone(),
            };
            self.test.extend(read_parquet_items(path, parse_item));
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        self.test.len() != POLYMATH_EXPECTED_LEN
            || self.test.iter().any(|item| {
                item.question.trim().is_empty()
                    || item.answer.trim().is_empty()
                    || item.id.trim().is_empty()
                    || !POLYMATH_LANGUAGES.contains(&item.language.as_str())
                    || !POLYMATH_SPLITS.contains(&item.difficulty.as_str())
            })
    }

    async fn download(&self) {
        let files = get_parquet_files(POLYMATH_DATASET)
            .await
            .into_iter()
            .filter(|file| {
                POLYMATH_LANGUAGES.contains(&file.config.as_str())
                    && POLYMATH_SPLITS.contains(&file.split.as_str())
            })
            .map(|file| UrlDownloadFile {
                relative_path: file.relative_path(),
                url: file.url,
            })
            .collect::<Vec<_>>();

        let downloaded_path = download_url_files(&self.dataset_root, "polymath", &files, 8).await;
        println!("polymath dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(n_shot, 0, "polymath only supports 0-shot");
        build_expected_context(&self.test[index], cot_mode)
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].answer.clone()
    }

    async fn answer_and_judge(
        &self,
        model_name: &str,
        model_client: &Client<OpenAIConfig>,
        judger_model_name: Option<&str>,
        judger_client: Option<&Client<OpenAIConfig>>,
        cot_mode: CoTMode,
        n_shot: u8,
        index: usize,
    ) -> Record {
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let generated = get_final_answer_with_cot_mode(
            model_client,
            model_name,
            &expected_context,
            &POLYMATH_INFO.sampling_config,
            cot_mode,
        )
        .await;
        let judger_client = judger_client
            .unwrap_or_else(|| panic!("benchmark requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("benchmark requires judger_model_name but got None"));

        let ref_answer = self.get_ref_answer(index);
        let parsed_answer = extract_boxed_answer(&generated.context)
            .or_else(|| extract_boxed_answer(&generated.answer))
            .unwrap_or_else(|| generated.answer.trim().to_string());
        let outcome = judge_with_retry(
            judger_client,
            judger_model_name,
            &expected_context,
            &ref_answer,
            &parsed_answer,
        )
        .await;

        Record {
            context: format!(
                "{}\n\n[boxed_answer]\n{}",
                generated.context, parsed_answer
            ),
            answer: parsed_answer,
            ref_answer,
            is_passed: outcome.is_passed,
            fail_reason: outcome.fail_reason,
        }
    }
}
