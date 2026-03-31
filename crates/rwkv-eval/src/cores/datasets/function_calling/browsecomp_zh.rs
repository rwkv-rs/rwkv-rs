use super::browsecomp_common::{
    BrowseCompLocale, browsecomp_sample_limit, build_browsecomp_expected_context,
    decrypt_xor_base64, generate_browsecomp_answer, judge_with_retry,
};
use crate::cores::datasets::utils::hf::downloader::{UrlDownloadFile, download_url_files};
use crate::cores::datasets::{
    ALL_BENCHMARKS, Benchmark, BenchmarkInfo, BenchmarkName, CoTMode, Field, Record, SamplingConfig,
};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use linkme::distributed_slice;
use regex::Regex;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use zip::ZipArchive;

const BROWSECOMP_ZH_EXPECTED_LEN: usize = 289;

#[distributed_slice(ALL_BENCHMARKS)]
static BROWSECOMP_ZH_INFO: BenchmarkInfo = BenchmarkInfo {
    name: BenchmarkName("browsecomp_zh"),
    field: Field::FunctionCalling,
    display_name: "BrowseComp-ZH",
    cot_mode: &[CoTMode::CoT],
    sampling_config: SamplingConfig {
        temperature: 0.3,
        top_k: 500,
        top_p: 0.4,
        presence_penalty: 0.5,
        repetition_penalty: 0.1,
        penalty_decay: 0.99,
    },
    n_shots: &[0],
    avg_ks: &[1.0],
    pass_ks: &[1],
    with_llm_judger: true,
    create: |dataset_root| Box::new(BrowseCompZh::new(dataset_root)),
};

pub struct BrowseCompZh {
    dataset_root: PathBuf,
    test: Vec<BrowseCompZhItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::cores::datasets::benchmark_dataset_tests!(BROWSECOMP_ZH_INFO);
}

pub struct BrowseCompZhItem {
    question: String,
    answer: String,
}

struct BrowseCompZhWireItem {
    question: String,
    answer: String,
}

impl BrowseCompZh {
    pub fn new<P: AsRef<Path>>(dataset_root: P) -> Self {
        Self {
            dataset_root: dataset_root.as_ref().to_path_buf(),
            test: Vec::new(),
        }
    }

    fn load_items(&self) -> Result<Vec<BrowseCompZhItem>, String> {
        let path = self
            .dataset_root
            .join("browsecomp_zh")
            .join("browsecomp-zh-encrypted.xlsx");
        if !path.is_file() {
            return Err(format!("missing browsecomp_zh xlsx: {}", path.display()));
        }

        load_browsecomp_zh_items_from_xlsx(&path)?
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                if row.question.trim().is_empty() || row.answer.trim().is_empty() {
                    return Err(format!("row {index} has empty question or answer"));
                }
                Ok(BrowseCompZhItem {
                    question: row.question,
                    answer: row.answer,
                })
            })
            .take(browsecomp_sample_limit().unwrap_or(usize::MAX))
            .collect()
    }
}

fn build_user_prompt(question: &str) -> String {
    format!(
        concat!(
            "你是一个浏览基准测试助手。请先仔细思考，再直接回答问题。\n\n",
            "请基于你自己的知识回答下面这个需要较强检索能力的问题。\n",
            "不要通过让用户自己去搜索网页来回避作答。\n",
            "即使你不完全确定，也要给出你当前最具体的答案。\n\n",
            "问题:\n{question}\n\n",
            "请严格按下面格式回复最终答案：\n",
            "解释: <简短说明>\n",
            "最终答案: <简洁最终答案>\n",
            "置信度: <0% 到 100%>"
        ),
        question = question
    )
}

#[async_trait]
impl Benchmark for BrowseCompZh {
    fn load(&mut self) -> bool {
        self.test.clear();

        let path = self
            .dataset_root
            .join("browsecomp_zh")
            .join("browsecomp-zh-encrypted.xlsx");
        if !path.is_file() {
            return true;
        }

        self.test = self
            .load_items()
            .unwrap_or_else(|err| panic!("failed to load browsecomp_zh: {err}"));
        self.test.is_empty()
    }

    async fn check(&self) -> bool {
        let size_invalid = if let Some(limit) = browsecomp_sample_limit() {
            self.test.len() != limit.min(BROWSECOMP_ZH_EXPECTED_LEN)
        } else {
            self.test.len() != BROWSECOMP_ZH_EXPECTED_LEN
        };

        size_invalid
            || self
                .test
                .iter()
                .any(|item| item.question.trim().is_empty() || item.answer.trim().is_empty())
    }

    async fn download(&self) {
        let downloaded_path = download_url_files(
            &self.dataset_root,
            "browsecomp_zh",
            &[UrlDownloadFile {
                relative_path: PathBuf::from("browsecomp-zh-encrypted.xlsx"),
                url: "https://raw.githubusercontent.com/PALIN2018/BrowseComp-ZH/main/data/browsecomp-zh-encrypted.xlsx".to_string(),
            }],
            1,
        )
        .await;
        println!("browsecomp_zh dataset: {}", downloaded_path.display());
    }

    fn len(&self) -> usize {
        self.test.len()
    }

    fn get_expected_context(&self, index: usize, cot_mode: CoTMode, n_shot: u8) -> String {
        assert_eq!(cot_mode, CoTMode::CoT, "browsecomp_zh only supports CoT");
        assert_eq!(n_shot, 0, "browsecomp_zh only supports 0-shot");

        build_browsecomp_expected_context(&build_user_prompt(&self.test[index].question))
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
        let item = &self.test[index];
        let expected_context = self.get_expected_context(index, cot_mode, n_shot);
        let ref_answer = self.get_ref_answer(index);
        let (context, answer) = generate_browsecomp_answer(
            model_client,
            model_name,
            &expected_context,
            &BROWSECOMP_ZH_INFO.sampling_config,
        )
        .await;

        if answer.trim().is_empty() {
            return Record {
                context,
                answer,
                ref_answer,
                is_passed: false,
                fail_reason: "model returned empty response".to_string(),
            };
        }

        let judger_client = judger_client
            .unwrap_or_else(|| panic!("browsecomp_zh requires judger_client but got None"));
        let judger_model_name = judger_model_name
            .unwrap_or_else(|| panic!("browsecomp_zh requires judger_model_name but got None"));
        let outcome = judge_with_retry(
            judger_client,
            judger_model_name,
            BrowseCompLocale::Zh,
            &item.question,
            &answer,
            &item.answer,
        )
        .await;

        Record {
            context,
            answer,
            ref_answer,
            is_passed: outcome.is_passed,
            fail_reason: if outcome.is_passed {
                String::new()
            } else if outcome.reason.is_empty() {
                "judger marked answer incorrect".to_string()
            } else {
                outcome.reason
            },
        }
    }
}

fn load_browsecomp_zh_items_from_xlsx(path: &Path) -> Result<Vec<BrowseCompZhWireItem>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open xlsx {}: {err}", path.display()))?;
    let mut archive =
        ZipArchive::new(file).map_err(|err| format!("failed to read xlsx zip: {err}"))?;
    let shared_strings =
        parse_shared_strings(&read_zip_entry(&mut archive, "xl/sharedStrings.xml")?)?;
    let rows = parse_named_sheet_rows(
        &read_zip_entry(&mut archive, "xl/worksheets/sheet1.xml")?,
        &shared_strings,
    )?;

    rows.into_iter()
        .enumerate()
        .map(|(index, row)| {
            let canary = row.get("canary").map(String::as_str).unwrap_or_default();
            let question = decrypt_optional_xlsx_field(row.get("Question"), canary)
                .map_err(|err| format!("row {index} question decrypt failed: {err}"))?;
            let answer = decrypt_optional_xlsx_field(row.get("Answer"), canary)
                .map_err(|err| format!("row {index} answer decrypt failed: {err}"))?;
            Ok(BrowseCompZhWireItem { question, answer })
        })
        .collect()
}

fn read_zip_entry<R: std::io::Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    name: &str,
) -> Result<String, String> {
    let mut entry = archive
        .by_name(name)
        .map_err(|err| format!("missing xlsx entry `{name}`: {err}"))?;
    let mut text = String::new();
    std::io::Read::read_to_string(&mut entry, &mut text)
        .map_err(|err| format!("failed to read xlsx entry `{name}`: {err}"))?;
    Ok(text)
}

fn decrypt_optional_xlsx_field(value: Option<&String>, password: &str) -> Result<String, String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    if value.trim().is_empty() {
        return Ok(String::new());
    }
    decrypt_xor_base64(value, password)
}

fn parse_shared_strings(xml: &str) -> Result<Vec<String>, String> {
    static SHARED_ITEM_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"(?s)<si\b[^>]*>(?P<body>.*?)</si>").unwrap());
    static TEXT_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r#"(?s)<t(?:\s+[^>]*)?>(?P<value>.*?)</t>"#).unwrap()
    });

    let values = SHARED_ITEM_RE
        .captures_iter(xml)
        .map(|caps| {
            TEXT_RE
                .captures_iter(caps.name("body").unwrap().as_str())
                .filter_map(|text_caps| text_caps.name("value").map(|m| m.as_str()))
                .map(xml_unescape)
                .collect::<Vec<_>>()
                .join("")
        })
        .collect::<Vec<_>>();

    if values.is_empty() {
        return Err("xlsx sharedStrings.xml did not contain any <si> entries".to_string());
    }

    Ok(values)
}

fn parse_named_sheet_rows(
    xml: &str,
    shared_strings: &[String],
) -> Result<Vec<BTreeMap<String, String>>, String> {
    static ROW_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"(?s)<row\b[^>]*>(?P<body>.*?)</row>").unwrap());
    static CELL_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r#"(?s)<c\b(?P<attrs>[^>]*)>(?P<body>.*?)</c>"#).unwrap()
    });
    static VALUE_RE: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"(?s)<v>(?P<value>.*?)</v>").unwrap());
    static TEXT_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(r#"(?s)<t(?:\s+[^>]*)?>(?P<value>.*?)</t>"#).unwrap()
    });

    let mut rows = Vec::new();
    let mut headers = BTreeMap::new();
    for row_caps in ROW_RE.captures_iter(xml) {
        let row_body = row_caps.name("body").unwrap().as_str();
        let mut cells_by_column = BTreeMap::new();
        for cell_caps in CELL_RE.captures_iter(row_body) {
            let attrs = cell_caps.name("attrs").unwrap().as_str();
            let body = cell_caps.name("body").unwrap().as_str();
            let Some(cell_ref) = extract_xml_attr(attrs, "r") else {
                continue;
            };
            let column = cell_column_name(cell_ref);
            let cell_type = extract_xml_attr(attrs, "t");
            let value = match cell_type {
                Some("s") => {
                    let raw_index = VALUE_RE
                        .captures(body)
                        .and_then(|caps| caps.name("value").map(|m| m.as_str().trim()))
                        .ok_or_else(|| format!("shared-string cell {cell_ref} missing <v>"))?;
                    let index = raw_index.parse::<usize>().map_err(|err| {
                        format!("invalid shared-string index `{raw_index}`: {err}")
                    })?;
                    shared_strings.get(index).cloned().ok_or_else(|| {
                        format!("shared-string index {index} out of range for cell {cell_ref}")
                    })?
                }
                Some("inlineStr") => TEXT_RE
                    .captures_iter(body)
                    .filter_map(|caps| caps.name("value").map(|m| xml_unescape(m.as_str())))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => VALUE_RE
                    .captures(body)
                    .and_then(|caps| caps.name("value").map(|m| xml_unescape(m.as_str())))
                    .unwrap_or_default(),
            };
            cells_by_column.insert(column, value);
        }

        if cells_by_column.is_empty() {
            continue;
        }

        if headers.is_empty() {
            headers = cells_by_column;
            continue;
        }

        let mut named_row = BTreeMap::new();
        for (column, value) in cells_by_column {
            if let Some(header) = headers
                .get(&column)
                .filter(|header| !header.trim().is_empty())
            {
                named_row.insert(header.clone(), value);
            }
        }
        if !named_row.is_empty() {
            rows.push(named_row);
        }
    }

    if headers.is_empty() {
        return Err("xlsx worksheet did not contain a header row".to_string());
    }

    Ok(rows)
}

fn extract_xml_attr<'a>(attrs: &'a str, name: &str) -> Option<&'a str> {
    let pattern = format!(r#"{name}=""#);
    let start = attrs.find(&pattern)? + pattern.len();
    let end = attrs[start..].find('"')?;
    Some(&attrs[start..start + end])
}

fn cell_column_name(cell_ref: &str) -> String {
    cell_ref
        .chars()
        .take_while(|ch| ch.is_ascii_alphabetic())
        .collect::<String>()
}

fn xml_unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}
