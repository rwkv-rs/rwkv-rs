use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use dashmap::DashSet;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use xxhash_rust::xxh3::xxh3_128;

use crate::processor::{Step, StepOutcome, file::Writer};

static HTML_PRE_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<pre[^>]*>.*?</pre>|<code[^>]*>.*?</code>").unwrap());

static CODE_FENCE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"```.*?```").unwrap());

#[derive(Clone)]
struct SentenceChunk {
    hash: u128,
    text: String,
}

impl SentenceChunk {
    fn new(text: String) -> Self {
        let hash = xxh3_128(text.as_bytes());

        Self { hash, text }
    }
}

fn compute_sentence_chunks(text: &str) -> Vec<SentenceChunk> {
    let mut protected_spans = Vec::new();

    for m in HTML_PRE_REGEX.find_iter(text) {
        protected_spans.push((m.start(), m.end()));
    }

    for m in CODE_FENCE_REGEX.find_iter(text) {
        let content = m.as_str();

        if content.chars().any(|c| c.is_ascii_alphabetic()) {
            protected_spans.push((m.start(), m.end()));
        }
    }

    let lines: Vec<&str> = text.lines().collect();

    let mut protected_lines = std::collections::HashSet::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed.matches('|').count() >= 2 {
            protected_lines.insert(i);
        }
    }

    let mut consecutive_indent = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("    ") && !line.trim().is_empty() {
            consecutive_indent.push(i);
        } else {
            if consecutive_indent.len() >= 3 {
                protected_lines.extend(consecutive_indent.iter());
            }

            consecutive_indent.clear();
        }
    }

    if consecutive_indent.len() >= 3 {
        protected_lines.extend(consecutive_indent.iter());
    }

    if !protected_lines.is_empty() {
        let mut pos = 0;

        for (i, line) in lines.iter().enumerate() {
            if protected_lines.contains(&i) {
                protected_spans.push((pos, pos + line.len()));
            }

            pos += line.len() + 1;
        }
    }

    protected_spans.sort_by(|a, b| a.0.cmp(&b.0));

    let mut merged: Vec<(usize, usize)> = Vec::new();

    for (start, end) in protected_spans {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        } else {
            merged.push((start, end));
        }
    }

    let mut chunks = Vec::new();

    if merged.is_empty() {
        split_into_chunks(text, &mut chunks);
    } else {
        let mut last_pos = 0;

        for (start, end) in merged {
            if start > last_pos {
                let unprotected = &text[last_pos..start];

                split_into_chunks(unprotected, &mut chunks);
            }

            let protected_content = text[start..end].trim();

            if !protected_content.is_empty() {
                chunks.push(SentenceChunk::new(protected_content.to_string()));
            }

            last_pos = end;
        }

        if last_pos < text.len() {
            let final_part = &text[last_pos..];

            split_into_chunks(final_part, &mut chunks);
        }
    }

    chunks
}

fn split_into_chunks(text: &str, chunks: &mut Vec<SentenceChunk>) {
    for line in text.lines() {
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        let mut current_sentence = String::new();

        let mut chars = line.chars().peekable();

        while let Some(ch) = chars.next() {
            current_sentence.push(ch);

            if ch == '.' || ch == '。' {
                let is_sentence_end = match chars.peek() {
                    Some(&next_ch) if next_ch.is_whitespace() || next_ch == '\n' => true,
                    None => true,
                    _ => false,
                };

                if is_sentence_end {
                    let sentence = current_sentence.trim();

                    if !sentence.is_empty() {
                        chunks.push(SentenceChunk::new(sentence.to_string()));
                    }

                    current_sentence.clear();

                    while chars.peek() == Some(&' ') || chars.peek() == Some(&'\t') {
                        chars.next();
                    }
                }
            }
        }

        let remaining = current_sentence.trim();

        if !remaining.is_empty() {
            chunks.push(SentenceChunk::new(remaining.to_string()));
        }
    }
}

pub struct ExactHashSentenceDedup {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
    seen_hashes: DashSet<u128>,
}

impl ExactHashSentenceDedup {
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self {
            writer,
            seen_hashes: DashSet::new(),
        }
    }
}

impl Step for ExactHashSentenceDedup {
    fn name(&self) -> &'static str {
        "ExactHashSentenceDedup"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        batch
            .into_par_iter()
            .map(|data| {
                let chunks = compute_sentence_chunks(data.as_ref());

                (data, chunks)
            })
            .map(|(data, chunks)| {
                let mut rebuilt = Vec::new();

                for chunk in chunks {
                    if self.seen_hashes.insert(chunk.hash) {
                        rebuilt.push(chunk.text);
                    }
                }

                if rebuilt.is_empty() {
                    StepOutcome::Exclude(data)
                } else {
                    StepOutcome::Keep(Cow::Owned(rebuilt.join("\n")))
                }
            })
            .collect()
    }
}
