use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    apply_user_assistant_template,
    instruction_following::sanitize_visible_answer,
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use crate::inferers::generate_text_completion;
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::Deserialize;
use std::path::{Path, PathBuf};

const WMT24PP_PASS_THRESHOLD: f64 = 70.0;
const WMT24PP_LANGUAGE_PAIRS: &[&str] = &[
    "en-ar_EG",
    "en-ar_SA",
    "en-bg_BG",
    "en-bn_IN",
    "en-ca_ES",
    "en-cs_CZ",
    "en-da_DK",
    "en-de_DE",
    "en-el_GR",
    "en-es_MX",
    "en-et_EE",
    "en-fa_IR",
    "en-fi_FI",
    "en-fil_PH",
    "en-fr_CA",
    "en-fr_FR",
    "en-gu_IN",
    "en-he_IL",
    "en-hi_IN",
    "en-hr_HR",
    "en-hu_HU",
    "en-id_ID",
    "en-is_IS",
    "en-it_IT",
    "en-ja_JP",
    "en-kn_IN",
    "en-ko_KR",
    "en-lt_LT",
    "en-lv_LV",
    "en-ml_IN",
    "en-mr_IN",
    "en-nl_NL",
    "en-no_NO",
    "en-pa_IN",
    "en-pl_PL",
    "en-pt_BR",
    "en-pt_PT",
    "en-ro_RO",
    "en-ru_RU",
    "en-sk_SK",
    "en-sl_SI",
    "en-sr_RS",
    "en-sv_SE",
    "en-sw_KE",
    "en-sw_TZ",
    "en-ta_IN",
    "en-te_IN",
    "en-th_TH",
    "en-tr_TR",
    "en-uk_UA",
    "en-ur_PK",
    "en-vi_VN",
    "en-zh_CN",
    "en-zh_TW",
    "en-zu_ZA",
];

#[distributed_slice(ALL_BENCHMARKS)]
static WMT24PP_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("wmt24pp"),
    field: Field::InstructionFollowing,
    display_name: "WMT24++",
    cot_mode: &[CoTMode::NoCoT],
    sampling_config: SamplingConfig {
        temperature: 0.001,
        top_k: 1,
        top_p: 1.0,
        presence_penalty: 0.0,
        repetition_penalty: 0.0,
        penalty_decay: 1.0,
    },
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(Wmt24pp::new(dataset_root)),
};
const WMT24PP_JUDGE_SAMPLING_CONFIG: SamplingConfig = SamplingConfig {
    temperature: 0.001,
    top_k: 1,
    top_p: 1.0,
    presence_penalty: 0.0,
    repetition_penalty: 0.0,
    penalty_decay: 1.0,
};
const WMT24PP_REQUEST_STOP_SUFFIXES: &[&str] = &["\n\nUser:"];
const WMT24PP_SANITIZE_STOP_SUFFIXES: &[&str] = &["\n\nUser:", "\n\nAssistant:"];
const WMT24PP_ASSISTANT_PREFIX: &str = "Translation:\n";
const WMT24PP_REFERENCE_BASED_JUDGE_INSTRUCTION: &str = concat!(
    "You are a professional translation quality evaluator.\n",
    "Evaluate the machine translation using the source text and the human reference translation.\n",
    "Assign a continuous score from 0 to 100, where 100 means the translation is excellent.\n",
    "Consider adequacy, fluency, terminology, and faithfulness.\n",
    "Reply with a single numeric score such as 83.5.\n",
    "Do not translate, copy, or rewrite any of the provided text.\n",
    "Output only the numeric score and nothing else."
);

pub struct Wmt24pp {
    dataset_root: PathBuf,
    test: Vec<Wmt24ppItem>,
}

pub struct Wmt24ppItem {
    lp: String,
    domain: String,
    document_id: String,
    segment_id: i64,
    source: String,
    target: String,
    original_target: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawWmt24ppItem {
    lp: String,
    domain: String,
    document_id: String,
    segment_id: i64,
    is_bad_source: bool,
    source: String,
    target: String,
    original_target: Option<String>,
}

#[derive(Debug)]
struct WmtJudgeOutcome {
    score: f64,
    reason: String,
}

impl Wmt24pp {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }
}

fn target_code_from_lp(lp: &str) -> &str {
    lp.split_once('-')
        .map(|(_, rhs)| rhs)
        .unwrap_or_else(|| panic!("invalid WMT24++ language pair `{lp}`"))
}

fn language_name_from_code(code: &str) -> &'static str {
    match code {
        "ar_EG" | "ar_SA" => "Arabic",
        "bg_BG" => "Bulgarian",
        "bn_IN" => "Bengali",
        "ca_ES" => "Catalan",
        "cs_CZ" => "Czech",
        "da_DK" => "Danish",
        "de_DE" => "German",
        "el_GR" => "Greek",
        "es_MX" => "Spanish",
        "et_EE" => "Estonian",
        "fa_IR" => "Farsi",
        "fi_FI" => "Finnish",
        "fil_PH" => "Filipino",
        "fr_CA" | "fr_FR" => "French",
        "gu_IN" => "Gujarati",
        "he_IL" => "Hebrew",
        "hi_IN" => "Hindi",
        "hr_HR" => "Croatian",
        "hu_HU" => "Hungarian",
        "id_ID" => "Indonesian",
        "is_IS" => "Icelandic",
        "it_IT" => "Italian",
        "ja_JP" => "Japanese",
        "kn_IN" => "Kannada",
        "ko_KR" => "Korean",
        "lt_LT" => "Lithuanian",
        "lv_LV" => "Latvian",
        "ml_IN" => "Malayalam",
        "mr_IN" => "Marathi",
        "nl_NL" => "Dutch",
        "no_NO" => "Norwegian",
        "pa_IN" => "Punjabi",
        "pl_PL" => "Polish",
        "pt_BR" | "pt_PT" => "Portuguese",
        "ro_RO" => "Romanian",
        "ru_RU" => "Russian",
        "sk_SK" => "Slovak",
        "sl_SI" => "Slovenian",
        "sr_RS" => "Serbian",
        "sv_SE" => "Swedish",
        "sw_KE" | "sw_TZ" => "Swahili",
        "ta_IN" => "Tamil",
        "te_IN" => "Telugu",
        "th_TH" => "Thai",
        "tr_TR" => "Turkish",
        "uk_UA" => "Ukrainian",
        "ur_PK" => "Urdu",
        "vi_VN" => "Vietnamese",
        "zh_CN" | "zh_TW" => "Mandarin",
        "zu_ZA" => "Zulu",
        _ => panic!("unsupported WMT24++ target code `{code}`"),
    }
}

fn region_name_from_code(code: &str) -> &'static str {
    match code {
        "ar_EG" => "Egypt",
        "ar_SA" => "Saudi Arabia",
        "bg_BG" => "Bulgaria",
        "bn_IN" => "India",
        "ca_ES" => "Spain",
        "cs_CZ" => "Czechia",
        "da_DK" => "Denmark",
        "de_DE" => "Germany",
        "el_GR" => "Greece",
        "es_MX" => "Mexico",
        "et_EE" => "Estonia",
        "fa_IR" => "Iran",
        "fi_FI" => "Finland",
        "fil_PH" => "Philippines",
        "fr_CA" => "Canada",
        "fr_FR" => "France",
        "gu_IN" => "India",
        "he_IL" => "Israel",
        "hi_IN" => "India",
        "hr_HR" => "Croatia",
        "hu_HU" => "Hungary",
        "id_ID" => "Indonesia",
        "is_IS" => "Iceland",
        "it_IT" => "Italy",
        "ja_JP" => "Japan",
        "kn_IN" => "India",
        "ko_KR" => "South Korea",
        "lt_LT" => "Lithuania",
        "lv_LV" => "Latvia",
        "ml_IN" => "India",
        "mr_IN" => "India",
        "nl_NL" => "Netherlands",
        "no_NO" => "Norway",
        "pa_IN" => "India",
        "pl_PL" => "Poland",
        "pt_BR" => "Brazil",
        "pt_PT" => "Portugal",
        "ro_RO" => "Romania",
        "ru_RU" => "Russia",
        "sk_SK" => "Slovakia",
        "sl_SI" => "Slovenia",
        "sr_RS" => "Serbia",
        "sv_SE" => "Sweden",
        "sw_KE" => "Kenya",
        "sw_TZ" => "Tanzania",
        "ta_IN" => "India",
        "te_IN" => "India",
        "th_TH" => "Thailand",
        "tr_TR" => "Turkey",
        "uk_UA" => "Ukraine",
        "ur_PK" => "Pakistan",
        "vi_VN" => "Vietnam",
        "zh_CN" => "China",
        "zh_TW" => "Taiwan",
        "zu_ZA" => "South Africa",
        _ => panic!("unsupported WMT24++ target code `{code}`"),
    }
}

fn build_translation_instruction(item: &Wmt24ppItem) -> String {
    let target_code = target_code_from_lp(&item.lp);
    let language = language_name_from_code(target_code);
    let region = region_name_from_code(target_code);

    format!(
        concat!(
            "You are a professional English to {language} translator.\n",
            "Your translation should be appropriate for {region} ({target_code}).\n",
            "Preserve the original meaning, subtle nuances, and tone while producing fluent, culturally appropriate {language}.\n",
            "Translate the following English text into {language}.\n",
            "Output only the translation and nothing else.\n\n",
            "{source}"
        ),
        language = language,
        region = region,
        target_code = target_code,
        source = item.source,
    )
}

fn generation_stop_suffixes() -> Vec<String> {
    WMT24PP_REQUEST_STOP_SUFFIXES
        .iter()
        .map(|suffix| (*suffix).to_string())
        .collect()
}

fn trim_transcript_contamination(text: &str) -> String {
    sanitize_visible_answer(text, WMT24PP_SANITIZE_STOP_SUFFIXES)
}

fn translation_max_tokens(source: &str) -> u32 {
    let source_word_count = source.split_whitespace().count();
    let dynamic_limit = source_word_count.saturating_mul(3).saturating_add(64);
    dynamic_limit.clamp(64, 1024) as u32
}

fn build_reference_based_judge_prompt(source: &str, reference: &str, translation: &str) -> String {
    format!(
        concat!(
            "Source text:\n{source}\n\n",
            "Human reference translation:\n{reference}\n\n",
            "Machine translation:\n{translation}\n"
        ),
        source = source,
        reference = reference,
        translation = translation,
    )
}

fn build_reference_based_judge_request(source: &str, reference: &str, translation: &str) -> String {
    format!(
        "{}\n\n{}",
        WMT24PP_REFERENCE_BASED_JUDGE_INSTRUCTION,
        build_reference_based_judge_prompt(source, reference, translation)
    )
}

fn extract_score(text: &str) -> Option<f64> {
    text.trim().parse::<f64>().ok()
}

async fn judge_once(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    source: &str,
    reference: &str,
    translation: &str,
) -> Result<WmtJudgeOutcome, String> {
    let prompt = build_reference_based_judge_request(source, reference, translation);
    let content = generate_text_completion(
        judger_client,
        judger_model_name,
        &prompt,
        vec![],
        64,
        &WMT24PP_JUDGE_SAMPLING_CONFIG,
    )
    .await
    .map_err(|err| format!("judge request failed: {err}"))?;
    let score = extract_score(&content)
        .ok_or_else(|| format!("judge returned no numeric score: {content:?}"))?;
    if !(0.0..=100.0).contains(&score) {
        return Err(format!(
            "judge returned out-of-range score {score}: {content:?}"
        ));
    }

    Ok(WmtJudgeOutcome {
        score,
        reason: content.trim().to_string(),
    })
}

async fn judge_with_retry(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    source: &str,
    reference: &str,
    translation: &str,
) -> Result<WmtJudgeOutcome, String> {
    for attempt in 1..=3 {
        match judge_once(
            judger_client,
            judger_model_name,
            source,
            reference,
            translation,
        )
        .await
        {
            Ok(outcome) => return Ok(outcome),
            Err(err) if attempt < 3 => {
                eprintln!(
                    "wmt24pp judge failed on attempt {attempt}/3, retrying: {err}; model={judger_model_name}"
                );
            }
            Err(err) => {
                return Err(format!(
                    "wmt24pp judge failed after 3 attempts: {err}; model={judger_model_name}; source={source}"
                ));
            }
        }
    }

    Err("unreachable wmt24pp judge retry loop".to_string())
}

#[async_trait]
impl Benchmark for Wmt24pp {
    fn load(&mut self) -> bool {
        self.test.clear();

        let paths = collect_files_with_extension(self.dataset_root.join("wmt24pp"), "jsonl");
        if paths.is_empty() {
            return true;
        }

        for path in paths {
            self.test.extend(
                read_jsonl_items::<RawWmt24ppItem, _>(&path)
                    .into_iter()
                    .filter(|row| !row.is_bad_source)
                    .map(|row| Wmt24ppItem {
                        lp: row.lp,
                        domain: row.domain,
                        document_id: row.document_id,
                        segment_id: row.segment_id,
                        source: row.source,
                        target: row.target,
                        original_target: row.original_target,
                    }),
            );
        }

        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        let file_count =
            collect_files_with_extension(self.dataset_root.join("wmt24pp"), "jsonl").len();

        file_count != WMT24PP_LANGUAGE_PAIRS.len()
            || self.test.is_empty()
            || self.test.iter().any(|item| {
                item.lp.trim().is_empty()
                    || item.domain.trim().is_empty()
                    || item.document_id.trim().is_empty()
                    || item.source.trim().is_empty()
                    || item.target.trim().is_empty()
            })
    }

    async fn download(&self) {
        let files = WMT24PP_LANGUAGE_PAIRS
            .iter()
            .map(|lp| UrlDownloadFile {
                relative_path: PathBuf::from(format!("{lp}.jsonl")),
                url: format!(
                    "https://huggingface.co/datasets/google/wmt24pp/resolve/main/{lp}.jsonl"
                ),
            })
            .collect::<Vec<_>>();

        let downloaded_path = download_url_files(&self.dataset_root, "wmt24pp", &files, 8).await;
        println!("wmt24pp dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(cot_mode, CoTMode::NoCoT, "wmt24pp only supports NoCoT");
        assert_eq!(n_shot, 0, "wmt24pp only supports 0-shot");
        apply_user_assistant_template(
            build_translation_instruction(&self.test[index]),
            WMT24PP_ASSISTANT_PREFIX.to_string(),
        )
    }

    fn get_ref_answer(&self, index: usize) -> String {
        self.test[index].target.clone()
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
        let item = &self.test[index];
        let prompt = self.get_expected_context(index, cot_mode, n_shot);
        let raw_answer = generate_text_completion(
            model_client,
            model_name,
            &prompt,
            generation_stop_suffixes(),
            translation_max_tokens(&item.source),
            &WMT24PP_INFO.sampling_config,
        )
        .await
        .unwrap();
        let answer = trim_transcript_contamination(&raw_answer);
        let ref_answer = self.get_ref_answer(index);
        let context = format!(
            concat!(
                "[lp]\n{}\n\n",
                "[domain]\n{}\n\n",
                "[document_id]\n{}\n\n",
                "[segment_id]\n{}\n\n",
                "[source]\n{}\n\n",
                "[reference_target]\n{}\n\n",
                "[original_target]\n{}\n\n",
                "[translation_prompt]\n{}\n\n",
                "[raw_model_translation]\n{}\n\n",
                "[model_translation]\n{}"
            ),
            item.lp,
            item.domain,
            item.document_id,
            item.segment_id,
            item.source,
            item.target,
            item.original_target.as_deref().unwrap_or(""),
            build_translation_instruction(item),
            raw_answer,
            answer,
        );

        if answer.trim().is_empty() {
            return Record {
                context,
                answer,
                ref_answer,
                is_passed: false,
                fail_reason: "model returned empty translation".to_string(),
            };
        }

        let judger_client =
            judger_client.unwrap_or_else(|| panic!("wmt24pp requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("wmt24pp requires judger_model_name but got None"));
        let outcome = match judge_with_retry(
            judger_client,
            judger_model_name,
            &item.source,
            &item.target,
            &answer,
        )
        .await
        {
            Ok(outcome) => outcome,
            Err(err) => {
                return Record {
                    context: format!("{context}\n\n[judge_error]\n{err}"),
                    answer,
                    ref_answer,
                    is_passed: false,
                    fail_reason: err,
                };
            }
        };
        let is_passed = outcome.score >= WMT24PP_PASS_THRESHOLD;

        Record {
            context: format!(
                "{}\n\n[judge]\nscore={:.2}\nthreshold={:.2}\n{}",
                context, outcome.score, WMT24PP_PASS_THRESHOLD, outcome.reason
            ),
            answer,
            ref_answer,
            is_passed,
            fail_reason: if is_passed {
                String::new()
            } else {
                format!(
                    "wmt24pp reference-based judge score {:.2} < threshold {:.2}",
                    outcome.score, WMT24PP_PASS_THRESHOLD
                )
            },
        }
    }
}
