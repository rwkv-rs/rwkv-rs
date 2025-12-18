//! Text Data Clean Pipeline
//!
//! 支持两种数据源的处理：
//! 1. GptOss: 从Parquet文件读取text字段，输出为{"text": "xxx"}的JSONL格式
//! 2. WildChat: 从Parquet文件读取conversation字段，转换为对话格式后输出为[{"😺"
//!    :"xxx"},{"🤖":"xxx"}]的JSONL格式

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use parquet::record::{Field, Row, RowAccessor};
use rwkv_data::processor::{
    Processor, Step,
    file::{
        reader::parquet::{FromParquetRow, ParquetReader},
        writer::json::JsonWriter,
    },
    pool::dedup::{
        exact_hash_sample::ExactHashSampleDedup, exact_hash_sentence::ExactHashSentenceDedup,
    },
    stream::{
        filter::{
            compression_ratio_tokenizer::TokenizerCompressionFilterStep,
            compression_ratio_zstd::ZstdCompressionFilterStep,
            language_char_script::LanguageCharScriptFilter,
            repetition_gopher::GopherRepetitionFilterStep,
        },
        formatter::{
            normalization::TextNormalizationFormatter,
            remove_special_token::RemoveSpecialTokenFormatter,
        },
    },
};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

/// 处理任务类型
enum Task {
    /// GPT-OSS数据处理：parquet(text) -> JSONL({"text": "xxx"})
    GptOss,
    /// WildChat数据处理：parquet(conversation) ->
    /// JSONL([{"😺":"xxx"},{"🤖":"xxx"}])
    WildChat,
}

/// GptOss数据结构 - 只有text字段
#[derive(Debug)]
struct GptOssRecord {
    text: String,
}

impl FromParquetRow for GptOssRecord {
    fn from_row(row: &Row) -> Self {
        let text = row.get_string(0).unwrap().to_string();

        Self { text }
    }
}

/// WildChat对话消息
#[derive(Debug, Deserialize)]
struct DialogueMessage {
    role: String,
    content: String,
}

/// WildChat数据结构
#[derive(Debug)]
struct WildChatRecord {
    conversation: Vec<DialogueMessage>,
}

impl FromParquetRow for WildChatRecord {
    fn from_row(row: &Row) -> Self {
        static CONV_IDX: OnceLock<usize> = OnceLock::new();

        let conversation_idx = *CONV_IDX.get_or_init(|| {
            row.get_column_iter()
                .enumerate()
                .find(|(_, (name, _))| name.as_str() == "conversation")
                .map(|(idx, _)| idx)
                .unwrap_or_else(|| {
                    let available_columns: Vec<String> = row
                        .get_column_iter()
                        .map(|(name, _)| name.to_string())
                        .collect();

                    panic!(
                        "WildChat record missing `conversation` column. Available columns: {}",
                        if available_columns.is_empty() {
                            "<none>".to_string()
                        } else {
                            available_columns.join(", ")
                        }
                    );
                })
        });

        let conversation_list = row.get_list(conversation_idx).unwrap_or_else(|err| {
            panic!("WildChat `conversation` column must be a repeated group, error: {err}");
        });

        let mut conversation = Vec::with_capacity(conversation_list.len());

        for element in conversation_list.elements() {
            let message_row = match element {
                Field::Group(group) => group,
                Field::Null => continue,
                other => panic!(
                    "WildChat `conversation` element must be a group, got {:?}",
                    other
                ),
            };

            if let Some(message) = DialogueMessage::from_parquet_row(message_row) {
                conversation.push(message);
            }
        }

        Self { conversation }
    }
}

impl DialogueMessage {
    fn from_parquet_row(row: &Row) -> Option<Self> {
        static IDX: OnceLock<(usize, usize)> = OnceLock::new();

        let (role_idx, content_idx) = *IDX.get_or_init(|| {
            let mut role = None;

            let mut content = None;

            for (idx, (name, _)) in row.get_column_iter().enumerate() {
                match name.as_str() {
                    "role" => role = Some(idx),
                    "content" => content = Some(idx),
                    _ => {},
                }
            }

            (
                role.unwrap_or_else(|| panic!("WildChat message row missing `role` column")),
                content.unwrap_or_else(|| panic!("WildChat message row missing `content` column")),
            )
        });

        let role = match row.get_string(role_idx) {
            Ok(value) => value.as_str().trim().to_owned(),
            Err(_) => return None,
        };

        let content = match row.get_string(content_idx) {
            Ok(value) => value.as_str().trim().to_owned(),
            Err(_) => return None,
        };

        if content.is_empty() && role.is_empty() {
            return None;
        }

        Some(Self { role, content })
    }
}

/// GptOss输出格式
#[derive(Serialize)]

struct GptOssOutput {
    text: String,
}

/// WildChat的对话条目 - 每个条目只包含一个字段
#[derive(Serialize)]
#[serde(untagged)]
enum DialogueEntry {
    User {
        #[serde(rename = "😺")]
        content: String,
    },
    Assistant {
        #[serde(rename = "🤖")]
        content: String,
    },
}

#[tokio::main]
async fn main() {
    let task = Task::WildChat;

    // println!("Found File: {}", get_parquet_files(Path::new(
    //     "/public/home/ssjxzkz/Datasets/lm/wildchat",
    // )).len());
    let exclusion = |data: Cow<'static, str>| -> Option<GptOssOutput> {
        let text = data.into_owned();

        if text.trim().is_empty() {
            None
        } else {
            Some(GptOssOutput { text })
        }
    };

    let mut tokenizer_filter = TokenizerCompressionFilterStep::new(Arc::new(JsonWriter::new(
        Path::new("./outputs/removed/tokenizer_compression"),
        exclusion,
    )));

    tokenizer_filter.set_vocab_path("./assets/rwkv_vocab_v20230424.txt".to_string());

    let tokenizer_filter: Arc<dyn Step> = Arc::new(tokenizer_filter);

    let steps: Vec<Arc<dyn Step>> = vec![
        Arc::new(ExactHashSampleDedup::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/exact_hash_sample"),
            exclusion,
        )))),
        Arc::new(RemoveSpecialTokenFormatter::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/remove_special_token"),
            exclusion,
        )))),
        Arc::new(TextNormalizationFormatter::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/text_normalization_1"),
            exclusion,
        )))),
        Arc::new(LanguageCharScriptFilter::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/language_char_script"),
            exclusion,
        )))),
        Arc::new(GopherRepetitionFilterStep::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/gopher_repetition"),
            exclusion,
        )))),
        Arc::new(ZstdCompressionFilterStep::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/zstd_compression"),
            exclusion,
        )))),
        tokenizer_filter,
        Arc::new(ExactHashSentenceDedup::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/exact_hash_sentence"),
            exclusion,
        )))),
        Arc::new(TextNormalizationFormatter::new(Arc::new(JsonWriter::new(
            Path::new("./outputs/removed/text_normalization_2"),
            exclusion,
        )))),
    ];

    match task {
        Task::GptOss => {
            let reader = ParquetReader::new(
                get_parquet_files(Path::new(
                    "/public/home/ssjxzkz/Datasets/lm/gpt-oss-20b-samples",
                )),
                // 转换器：直接返回text字段
                |record: GptOssRecord| -> Cow<'static, str> { Cow::Owned(record.text) },
            );

            let writer = JsonWriter::new(
                Path::new("./outputs/final"),
                // 转换器：包装为{"text": "xxx"}格式
                |data: Cow<'static, str>| -> Option<GptOssOutput> {
                    let text = data.into_owned();

                    if text.trim().is_empty() {
                        None
                    } else {
                        Some(GptOssOutput { text })
                    }
                },
            );

            let processor = Processor::new(reader, steps, writer);

            processor.run().await;

            println!("GptOss processing completed!");
        },
        Task::WildChat => {
            let reader = ParquetReader::new(
                get_parquet_files(Path::new("/public/home/ssjxzkz/Datasets/lm/wildchat")),
                // 转换器：拼接对话为字符串
                |record: WildChatRecord| -> Cow<'static, str> {
                    let dialogue_text = record
                        .conversation
                        .into_iter()
                        .filter_map(|msg| {
                            if msg.content.trim().is_empty() {
                                None
                            } else if msg.role == "user" {
                                Some(format!("😺: {}", msg.content.trim()))
                            } else if msg.role == "assistant" {
                                Some(format!("🤖: {}", msg.content.trim()))
                            } else {
                                None // 忽略其他角色
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    Cow::Owned(dialogue_text)
                },
            );

            let writer = JsonWriter::new(
                Path::new("./outputs/final"),
                // 转换器：解析字符串重建对话数组
                |data: Cow<'static, str>| -> Option<Vec<DialogueEntry>> {
                    let mut raw_entries: Vec<DialogueEntry> = data
                        .lines()
                        .filter_map(|line| {
                            if let Some(content) = line.strip_prefix("😺: ") {
                                Some(DialogueEntry::User {
                                    content: content.to_string(),
                                })
                            } else if let Some(content) = line.strip_prefix("🤖: ") {
                                Some(DialogueEntry::Assistant {
                                    content: content.to_string(),
                                })
                            } else {
                                None
                            }
                        })
                        .collect();

                    if raw_entries.is_empty() {
                        return None;
                    }

                    if let Some(first_user_idx) = raw_entries
                        .iter()
                        .position(|entry| matches!(entry, DialogueEntry::User { .. }))
                    {
                        if first_user_idx > 0 {
                            raw_entries.drain(0..first_user_idx);
                        }
                    } else {
                        return None;
                    }

                    let mut cleaned = Vec::with_capacity(raw_entries.len());

                    let mut expect_user = true;

                    for entry in raw_entries.into_iter() {
                        match entry {
                            DialogueEntry::User { .. } if expect_user => {
                                cleaned.push(entry);

                                expect_user = false;
                            },
                            DialogueEntry::User { .. } => {
                                if matches!(cleaned.last(), Some(DialogueEntry::User { .. })) {
                                    cleaned.pop();
                                }

                                cleaned.push(entry);

                                expect_user = false;
                            },
                            DialogueEntry::Assistant { .. } if !expect_user => {
                                cleaned.push(entry);

                                expect_user = true;
                            },
                            DialogueEntry::Assistant { .. } => {
                                continue;
                            },
                        }
                    }

                    if matches!(cleaned.last(), Some(DialogueEntry::User { .. })) {
                        cleaned.pop();
                    }

                    if cleaned.len() < 2 {
                        None
                    } else {
                        Some(cleaned)
                    }
                },
            );

            let processor = Processor::new(reader, steps, writer);

            processor.run().await;

            println!("WildChat processing completed!");
        },
    }
}

fn get_parquet_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok()) // 过滤掉出错的项
        .filter(|entry| {
            entry.file_type().is_file()
                && entry
                    .path()
                    .extension()
                    .map_or(false, |ext| ext == "parquet")
        })
        .map(|entry| entry.path().to_path_buf())
        .collect()
}
