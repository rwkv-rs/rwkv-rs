use crate::datasets::maths::{
    extract_last_boxed_answer, get_final_answer_with_cot_mode, judge_with_retry,
};
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

fn get_solution_note(language: &str) -> &'static str {
    match language {
        "en" => "Solve the problem carefully step by step.",
        "zh" => "请认真逐步求解此题。",
        "ar" => "حل المسألة بعناية خطوة بخطوة.",
        "bn" => "প্রশ্নটি ধাপে ধাপে সতর্কভাবে সমাধান করুন।",
        "de" => "Losen Sie das Problem sorgfaltig Schritt fur Schritt.",
        "es" => "Resuelve el problema con cuidado paso a paso.",
        "fr" => "Resolvez le probleme soigneusement etape par etape.",
        "id" => "Selesaikan soal ini dengan cermat langkah demi langkah.",
        "it" => "Risolvi il problema con attenzione, passo dopo passo.",
        "ja" => "問題を丁寧に一歩ずつ解いてください。",
        "ko" => "문제를 차근차근 신중하게 풀어 주세요.",
        "ms" => "Selesaikan masalah ini dengan teliti langkah demi langkah.",
        "pt" => "Resolva o problema com cuidado, passo a passo.",
        "ru" => "Решите задачу внимательно, шаг за шагом.",
        "sw" => "Tatua tatizo hili kwa makini hatua kwa hatua.",
        "te" => "ఈ సమస్యను జాగ్రత్తగా దశలవారీగా పరిష్కరించండి.",
        "th" => "แก้โจทย์นี้อย่างรอบคอบทีละขั้นตอน",
        "vi" => "Hãy giải bài toán cẩn thận từng bước một.",
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

fn normalize_reference_answer(answer: &str) -> String {
    extract_last_boxed_answer(answer).unwrap_or_else(|| answer.trim().to_string())
}

fn build_expected_context(item: &PolyMathItem, cot_mode: CoTMode) -> String {
    if cot_mode != CoTMode::CoT {
        panic!("polymath only supports CoT mode, got {cot_mode:?}");
    }

    let user_part = format!(
        "{}\n\n{}\n{}\n{}",
        item.question.trim(),
        get_language_control(&item.language),
        get_solution_note(&item.language),
        get_answer_note(&item.language),
    );

    let assistant_part = concat!(
        "<think><|completions_of_cot|></think>\n",
        "Therefore, the answer is \\(\\boxed{<|final_answer|>}\\)."
    )
    .to_string();

    apply_user_assistant_template(user_part, assistant_part)
}

const POLYMATH_ASSISTANT_MARKER: &str = "\n\nAssistant: ";
const POLYMATH_BOXED_ANSWER_MARKER: &str = "\n\n[boxed_answer]\n";

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolyMathFailureBucket {
    ReasoningWrongFormatOk,
    BoxedPresentButWrong,
    BoxedMissing,
    BoxedPresentButExtractionMismatch,
    PromptOrTranscriptContamination,
    LanguageDrift,
    OtherGenerationArtifact,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolyMathAttemptDiagnostics {
    pub generated_answer: String,
    pub transcript_contamination: bool,
    pub think_leakage: bool,
    pub parsed_boxed_answer: Option<String>,
    pub stored_answer_matches_parsed_boxed_answer: bool,
    pub bucket: PolyMathFailureBucket,
}

pub fn extract_generated_answer_from_record_context(context: &str) -> Option<&str> {
    if let Some((with_prompt, _)) = context.rsplit_once(POLYMATH_BOXED_ANSWER_MARKER) {
        if let Some((_, generated_answer)) = with_prompt.split_once(POLYMATH_ASSISTANT_MARKER) {
            return Some(generated_answer);
        }
    }

    context
        .split_once(POLYMATH_ASSISTANT_MARKER)
        .map(|(_, generated_answer)| generated_answer)
}

pub fn contains_transcript_contamination(text: &str) -> bool {
    text.contains("\nUser:")
        || text.contains("\nAssistant:")
        || text.trim_start().starts_with("User:")
        || text.trim_start().starts_with("Assistant:")
}

pub fn contains_think_leakage(text: &str) -> bool {
    let open = text.find("<think>");
    let close = text.rfind("</think>");

    match (open, close) {
        (None, None) => false,
        (Some(_), None) | (None, Some(_)) => true,
        (Some(open), Some(close)) if close < open => true,
        (Some(_), Some(close)) => match text.rfind(r"\boxed{") {
            Some(boxed_pos) => close > boxed_pos,
            None => false,
        },
    }
}

pub fn diagnose_failed_attempt(
    context: &str,
    stored_answer: &str,
    ref_answer: &str,
) -> PolyMathAttemptDiagnostics {
    let generated_answer = extract_generated_answer_from_record_context(context)
        .unwrap_or_default()
        .to_string();
    let transcript_contamination = contains_transcript_contamination(&generated_answer);
    let think_leakage = contains_think_leakage(&generated_answer);
    let parsed_boxed_answer = extract_last_boxed_answer(&generated_answer);
    let stored_answer = stored_answer.trim();
    let ref_answer = ref_answer.trim();
    let stored_answer_matches_parsed_boxed_answer = parsed_boxed_answer
        .as_deref()
        .map(str::trim)
        .is_some_and(|parsed| parsed == stored_answer);

    let bucket = if generated_answer.is_empty() {
        PolyMathFailureBucket::OtherGenerationArtifact
    } else if transcript_contamination {
        PolyMathFailureBucket::PromptOrTranscriptContamination
    } else if let Some(parsed_boxed_answer) = parsed_boxed_answer.as_deref().map(str::trim) {
        if stored_answer.is_empty() || parsed_boxed_answer != stored_answer {
            PolyMathFailureBucket::BoxedPresentButExtractionMismatch
        } else if stored_answer != ref_answer {
            PolyMathFailureBucket::BoxedPresentButWrong
        } else {
            PolyMathFailureBucket::ReasoningWrongFormatOk
        }
    } else if think_leakage {
        PolyMathFailureBucket::OtherGenerationArtifact
    } else if stored_answer.is_empty() {
        PolyMathFailureBucket::BoxedMissing
    } else if stored_answer != ref_answer {
        PolyMathFailureBucket::ReasoningWrongFormatOk
    } else {
        PolyMathFailureBucket::OtherGenerationArtifact
    };

    PolyMathAttemptDiagnostics {
        generated_answer,
        transcript_contamination,
        think_leakage,
        parsed_boxed_answer,
        stored_answer_matches_parsed_boxed_answer,
        bucket,
    }
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
        normalize_reference_answer(&self.test[index].answer)
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
        let judge_answer = if generated.answer.trim().is_empty() {
            extract_generated_answer_from_record_context(&generated.context)
                .unwrap_or_default()
                .trim()
                .to_string()
        } else {
            generated.answer.trim().to_string()
        };
        let outcome = judge_with_retry(
            judger_client,
            judger_model_name,
            &expected_context,
            &ref_answer,
            &judge_answer,
        )
        .await;

        Record {
            context: generated.context,
            answer: generated.answer,
            ref_answer,
            is_passed: outcome.is_passed,
            fail_reason: outcome.fail_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PolyMathFailureBucket, PolyMathItem, build_expected_context, contains_think_leakage,
        contains_transcript_contamination, diagnose_failed_attempt,
        extract_generated_answer_from_record_context, normalize_reference_answer,
    };
    use crate::datasets::CoTMode;

    #[test]
    fn normalizes_boxed_reference_answer() {
        let raw = "### الشرح:\n...\nإذًا الجواب هو \\boxed{117}.";
        assert_eq!(normalize_reference_answer(raw), "117");
    }

    #[test]
    fn preserves_plain_reference_answer() {
        assert_eq!(normalize_reference_answer("588"), "588");
    }

    #[test]
    fn normalizes_nested_boxed_reference_answer() {
        let raw = r"Final: \boxed{\frac{13}{4}}";
        assert_eq!(normalize_reference_answer(raw), r"\frac{13}{4}");
    }

    #[test]
    fn builds_two_stage_prompt_with_language_control_and_boxed_answer() {
        let item = PolyMathItem {
            id: "1".to_string(),
            question: "Q".to_string(),
            answer: "A".to_string(),
            language: "ar".to_string(),
            difficulty: "medium".to_string(),
        };
        let prompt = build_expected_context(&item, CoTMode::CoT);

        assert!(prompt.starts_with("User: Q"));
        assert!(prompt.contains("استخدم العربية في التفكير والإجابة."));
        assert!(prompt.contains("حل المسألة بعناية خطوة بخطوة."));
        assert!(prompt.contains("ضع الإجابة النهائية داخل \\boxed{}."));
        assert!(prompt.contains("<think><|completions_of_cot|></think>"));
        assert!(prompt.contains(r"\boxed{<|final_answer|>}"));
    }

    #[test]
    fn extracts_generated_answer_from_two_stage_context() {
        let context =
            "User: Q\n\nAssistant: <think>step 1</think>\nTherefore, the answer is \\(\\boxed{42}\\).";
        assert_eq!(
            extract_generated_answer_from_record_context(context),
            Some("<think>step 1</think>\nTherefore, the answer is \\(\\boxed{42}\\).")
        );
    }

    #[test]
    fn detects_transcript_contamination_in_generated_answer() {
        let generated = "Reasoning...\nUser: new prompt";
        assert!(contains_transcript_contamination(generated));
    }

    #[test]
    fn detects_think_leakage_in_generated_answer() {
        let generated = "<think>hidden chain";
        assert!(contains_think_leakage(generated));
    }

    #[test]
    fn classifies_transcript_contamination_before_other_failures() {
        let context = concat!(
            "User: Q\n\nAssistant: final is \\boxed{41}\nUser: new prompt",
            "\n\n[boxed_answer]\n41"
        );
        let diagnostics = diagnose_failed_attempt(context, "41", "42");
        assert_eq!(
            diagnostics.bucket,
            PolyMathFailureBucket::PromptOrTranscriptContamination
        );
        assert!(diagnostics.transcript_contamination);
    }

    #[test]
    fn classifies_unfinished_think_without_boxed_answer_as_generation_artifact() {
        let context = "User: Q\n\nAssistant: <think>draft";
        let diagnostics = diagnose_failed_attempt(context, "", "42");
        assert_eq!(
            diagnostics.bucket,
            PolyMathFailureBucket::OtherGenerationArtifact
        );
        assert!(diagnostics.think_leakage);
    }

    #[test]
    fn classifies_boxed_answer_extraction_mismatch() {
        let context =
            "User: Q\n\nAssistant: <think>draft</think>\nTherefore, the answer is \\(\\boxed{\\frac{13}{4}}\\).";
        let diagnostics = diagnose_failed_attempt(context, "13/4", r"\frac{13}{4}");
        assert_eq!(
            diagnostics.bucket,
            PolyMathFailureBucket::BoxedPresentButExtractionMismatch
        );
        assert_eq!(
            diagnostics.parsed_boxed_answer.as_deref(),
            Some(r"\frac{13}{4}")
        );
        assert!(!diagnostics.stored_answer_matches_parsed_boxed_answer);
    }

    #[test]
    fn classifies_boxed_present_but_wrong() {
        let context =
            "User: Q\n\nAssistant: <think>draft</think>\nTherefore, the answer is \\(\\boxed{41}\\).";
        let diagnostics = diagnose_failed_attempt(context, "41", "42");
        assert_eq!(
            diagnostics.bucket,
            PolyMathFailureBucket::BoxedPresentButWrong
        );
        assert!(diagnostics.stored_answer_matches_parsed_boxed_answer);
        assert!(!diagnostics.think_leakage);
    }

    #[test]
    fn classifies_boxed_missing_when_stored_answer_is_empty() {
        let context = "User: Q\n\nAssistant: <think>draft</think>\nThe answer is 41.";
        let diagnostics = diagnose_failed_attempt(context, "", "42");
        assert_eq!(diagnostics.bucket, PolyMathFailureBucket::BoxedMissing);
        assert_eq!(diagnostics.parsed_boxed_answer, None);
    }

    #[test]
    fn classifies_reasoning_wrong_when_format_is_otherwise_ok() {
        let context = "User: Q\n\nAssistant: The answer is 41.";
        let diagnostics = diagnose_failed_attempt(context, "41", "42");
        assert_eq!(
            diagnostics.bucket,
            PolyMathFailureBucket::ReasoningWrongFormatOk
        );
        assert_eq!(diagnostics.parsed_boxed_answer, None);
    }
}
