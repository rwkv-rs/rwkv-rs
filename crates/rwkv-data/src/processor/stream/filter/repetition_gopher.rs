use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::Arc,
};

use rayon::prelude::*;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;
use xxhash_rust::xxh3::xxh3_128;

use crate::processor::{Step, StepOutcome, file::Writer};

const SHORT_TEXT_THRESHOLD: usize = 512;

fn find_duplicates(items: &[&str]) -> (usize, usize) {
    let mut unique_items: HashSet<&str> = HashSet::with_capacity(items.len());

    let mut duplicate_chars = 0;

    let mut duplicate_elements = 0;

    for &item in items {
        if !unique_items.insert(item) {
            duplicate_chars += item.len();

            duplicate_elements += 1;
        }
    }

    (duplicate_elements, duplicate_chars)
}

fn split_into_words(text: &str) -> Vec<&str> {
    UnicodeSegmentation::unicode_words(text).collect()
}

fn top_ngram_duplicate(words: &[&str], n: usize, buffer: &mut Vec<u8>) -> usize {
    if words.len() < n {
        return 0;
    }

    let mut counter: HashMap<u128, (usize, usize)> = HashMap::new();

    for window in words.windows(n) {
        buffer.clear();

        let mut total_len = 0;

        for (idx, word) in window.iter().enumerate() {
            buffer.extend_from_slice(word.as_bytes());

            total_len += word.len();

            if idx + 1 < window.len() {
                buffer.push(b' ');

                total_len += 1; // 计算 join(" ") 后的空格
            }
        }

        let hash = xxh3_128(&buffer);

        let entry = counter.entry(hash).or_insert((0, total_len));

        entry.0 += 1;
    }

    counter
        .values()
        .map(|(count, item_len)| count * item_len)
        .max()
        .unwrap_or(0)
}

fn repeated_ngram_chars(words: &[&str], n: usize, buffer: &mut Vec<u8>) -> usize {
    if words.len() < n {
        return 0;
    }

    let mut unique: HashSet<u128> = HashSet::with_capacity(words.len());

    let mut repeated_chars = 0;

    let mut idx = 0;

    while idx + n <= words.len() {
        buffer.clear();

        let mut total_len = 0;

        for word in &words[idx..idx + n] {
            buffer.extend_from_slice(word.as_bytes());

            total_len += word.len();
        }

        let hash = xxh3_128(&buffer);

        if !unique.insert(hash) {
            repeated_chars += total_len;

            idx += n;
        } else {
            idx += 1;
        }
    }

    repeated_chars
}

struct GopherRepetitionFilter {
    dup_line_frac: Option<f64>,
    dup_para_frac: Option<f64>,
    dup_line_char_frac: Option<f64>,
    dup_para_char_frac: Option<f64>,
    top_n_grams: Vec<(usize, f64)>,
    dup_n_grams: Vec<(usize, f64)>,
    paragraph_regex: Regex,
    line_regex: Regex,
}

impl GopherRepetitionFilter {
    fn new() -> Self {
        let paragraph_regex = Regex::new(r"\n{2,}").unwrap();

        let line_regex = Regex::new(r"\n+").unwrap();

        Self {
            dup_line_frac: Some(0.3),
            dup_para_frac: Some(0.3),
            dup_line_char_frac: Some(0.2),
            dup_para_char_frac: Some(0.2),
            top_n_grams: vec![(2, 0.2), (3, 0.18), (4, 0.16)],
            dup_n_grams: vec![
                (5, 0.15),
                (6, 0.14),
                (7, 0.13),
                (8, 0.12),
                (9, 0.11),
                (10, 0.10),
            ],
            paragraph_regex,
            line_regex,
        }
    }

    fn should_filter(&self, text: &str) -> bool {
        let trimmed = text.trim();

        if trimmed.is_empty() {
            return true;
        }

        let text_len = trimmed.len();

        // Check paragraph duplicates
        let paragraphs: Vec<&str> = self
            .paragraph_regex
            .split(trimmed)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if !paragraphs.is_empty() {
            let (paragraphs_duplicates, para_char_duplicates) = find_duplicates(&paragraphs);

            if let Some(threshold) = self.dup_para_frac {
                if paragraphs_duplicates as f64 / paragraphs.len() as f64 > threshold {
                    return true;
                }
            }

            if let Some(threshold) = self.dup_para_char_frac {
                if para_char_duplicates as f64 / text_len as f64 > threshold {
                    return true;
                }
            }
        }

        // Check line duplicates
        let lines: Vec<&str> = self
            .line_regex
            .split(trimmed)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        if !lines.is_empty() {
            let (line_duplicates, line_char_duplicates) = find_duplicates(&lines);

            if let Some(threshold) = self.dup_line_frac {
                if line_duplicates as f64 / lines.len() as f64 > threshold {
                    return true;
                }
            }

            if let Some(threshold) = self.dup_line_char_frac {
                if line_char_duplicates as f64 / text_len as f64 > threshold {
                    return true;
                }
            }
        }

        if text_len < SHORT_TEXT_THRESHOLD {
            // 短文本直接放行，重复度统计意义不大
            return false;
        }

        let words = split_into_words(trimmed);

        if words.len() < 2 {
            return false;
        }

        let mut hash_buffer = Vec::with_capacity(256);

        // Check top n-grams
        for &(n, n_frac) in &self.top_n_grams {
            let top_char_length = top_ngram_duplicate(&words, n, &mut hash_buffer);

            if top_char_length as f64 / text_len as f64 > n_frac {
                return true;
            }
        }

        // Check duplicate n-grams
        for &(n, n_frac) in &self.dup_n_grams {
            let n_duplicates_char = repeated_ngram_chars(&words, n, &mut hash_buffer);

            if n_duplicates_char as f64 / text_len as f64 > n_frac {
                return true;
            }
        }

        false
    }
}

pub struct GopherRepetitionFilterStep {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
    filter: GopherRepetitionFilter,
}

impl GopherRepetitionFilterStep {
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self {
            writer,
            filter: GopherRepetitionFilter::new(),
        }
    }
}

impl Step for GopherRepetitionFilterStep {
    fn name(&self) -> &'static str {
        "GopherRepetitionFilter"
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
                if self.filter.should_filter(data.as_ref()) {
                    StepOutcome::Exclude(data)
                } else {
                    StepOutcome::Keep(data)
                }
            })
            .collect()
    }
}
