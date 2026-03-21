use crate::datasets::utils::collect_files_with_extension;
use crate::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::datasets::utils::jsonl::read_jsonl_items;
use crate::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use crate::evaluators::instruction_following::{build_prompt, generate_response};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use serde::{Deserialize, Serialize};
use sonic_rs::{Value, json};
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
        temperature: 0.0,
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
    original_target: String,
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
    original_target: String,
}

#[derive(Debug, Serialize)]
struct WmtJudgeRequest<'a> {
    model: &'a str,
    messages: Vec<WmtJudgeMessage<'a>>,
    temperature: f32,
    top_p: f32,
    max_completion_tokens: u32,
    response_format: WmtJudgeResponseFormat,
}

#[derive(Debug, Serialize)]
struct WmtJudgeMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct WmtJudgeResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: WmtJudgeJsonSchema,
}

#[derive(Debug, Serialize)]
struct WmtJudgeJsonSchema {
    description: Option<String>,
    name: String,
    schema: Value,
    strict: bool,
}

#[derive(Debug, Deserialize)]
struct WmtJudgeResponse {
    choices: Vec<WmtJudgeChoice>,
}

#[derive(Debug, Deserialize)]
struct WmtJudgeChoice {
    finish_reason: Option<String>,
    message: WmtJudgeChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct WmtJudgeChoiceMessage {
    content: Option<String>,
    refusal: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct WmtJudgeWire {
    score: f64,
    reason: String,
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

async fn judge_once(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    source: &str,
    reference: &str,
    translation: &str,
) -> Result<WmtJudgeOutcome, String> {
    let user_prompt = build_reference_based_judge_prompt(source, reference, translation);
    let req = WmtJudgeRequest {
        model: judger_model_name,
        messages: vec![
            WmtJudgeMessage {
                role: "system",
                content: concat!(
                    "You are a professional translation quality evaluator.\n",
                    "Evaluate the machine translation using the source text and the human reference translation.\n",
                    "Assign a continuous score from 0 to 100, where 100 means the translation is excellent.\n",
                    "Consider adequacy, fluency, terminology, and faithfulness.\n",
                    "Return only JSON matching the provided schema."
                ),
            },
            WmtJudgeMessage {
                role: "user",
                content: &user_prompt,
            },
        ],
        temperature: 0.0,
        top_p: 1.0,
        max_completion_tokens: 256,
        response_format: WmtJudgeResponseFormat {
            kind: "json_schema",
            json_schema: WmtJudgeJsonSchema {
                description: None,
                name: "wmt24pp_reference_based_judge".to_string(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 100.0
                        },
                        "reason": { "type": "string" }
                    },
                    "required": ["score", "reason"],
                    "additionalProperties": false
                }),
                strict: true,
            },
        },
    };

    let resp: WmtJudgeResponse = judger_client
        .chat()
        .create_byot(&req)
        .await
        .map_err(|err| format!("judge request failed: {err}"))?;
    let choice = resp
        .choices
        .first()
        .ok_or_else(|| "judge returned no choices".to_string())?;
    let content = choice.message.content.clone().ok_or_else(|| {
        format!(
            "judge returned no content; refusal={:?}; finish_reason={:?}",
            choice.message.refusal, choice.finish_reason
        )
    })?;
    let parsed = sonic_rs::from_str::<WmtJudgeWire>(&content)
        .map_err(|err| format!("invalid judge json: {err}; content={content:?}"))?;

    Ok(WmtJudgeOutcome {
        score: parsed.score,
        reason: parsed.reason.trim().to_string(),
    })
}

async fn judge_with_retry(
    judger_client: &Client<OpenAIConfig>,
    judger_model_name: &str,
    source: &str,
    reference: &str,
    translation: &str,
) -> WmtJudgeOutcome {
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
            Ok(outcome) => return outcome,
            Err(err) if attempt < 3 => {
                eprintln!(
                    "wmt24pp judge failed on attempt {attempt}/3, retrying: {err}; model={judger_model_name}"
                );
            }
            Err(err) => {
                panic!(
                    "wmt24pp judge failed after 3 attempts: {err}; model={judger_model_name}; source={source}"
                );
            }
        }
    }

    panic!("unreachable wmt24pp judge retry loop")
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
        build_prompt(&build_translation_instruction(&self.test[index]))
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
        let answer = generate_response(
            model_client,
            model_name,
            &prompt,
            &WMT24PP_INFO.sampling_config,
        )
        .await
        .trim()
        .to_string();
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
                "[model_translation]\n{}"
            ),
            item.lp,
            item.domain,
            item.document_id,
            item.segment_id,
            item.source,
            item.target,
            item.original_target,
            build_translation_instruction(item),
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
        let outcome = judge_with_retry(
            judger_client,
            judger_model_name,
            &item.source,
            &item.target,
            &answer,
        )
        .await;
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

#[cfg(test)]
mod tests {
    use super::{language_name_from_code, region_name_from_code, target_code_from_lp};

    #[test]
    fn parses_language_pair_metadata() {
        assert_eq!(target_code_from_lp("en-de_DE"), "de_DE");
        assert_eq!(language_name_from_code("de_DE"), "German");
        assert_eq!(region_name_from_code("de_DE"), "Germany");
    }
}
