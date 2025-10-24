use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use rayon::prelude::*;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use unicode_script::{Script, UnicodeScript};

use crate::processor::{Step, StepOutcome, file::Writer};

struct TextNormalizer {
    html_pre_regex: Regex,
    code_fence_regex: Regex,
    zero_width_regex: Regex,
    unicode_space_regex: Regex,
    multiple_newline_regex: Regex,
    cjk_latin_space_regex: Regex,
    latin_cjk_space_regex: Regex,
    code_block_junk_regex: Regex,
    cjk_mode: bool,
}

impl TextNormalizer {
    fn new(cjk_mode: bool) -> Self {
        let html_pre_regex = Regex::new(r"(?is)<pre[^>]*>.*?</pre>|<code[^>]*>.*?</code>").unwrap();

        let code_fence_regex = Regex::new(r"```.*?```").unwrap();

        let zero_width_regex = Regex::new(r"[\u{200B}-\u{200D}\u{2060}\u{FEFF}]").unwrap();

        let unicode_space_regex =
            Regex::new(r"[\u{00A0}\u{1680}\u{2000}-\u{200A}\u{202F}\u{205F}\u{3000}]").unwrap();

        let multiple_newline_regex = Regex::new(r"\n{3,}").unwrap();

        let cjk_latin_space_regex = Regex::new(r"([\u{4e00}-\u{9fff}\u{3040}-\u{309f}\u{30a0}-\u{30ff}\u{ac00}-\u{d7af}])\s+([A-Za-z0-9])").unwrap();

        let latin_cjk_space_regex = Regex::new(r"([A-Za-z0-9])\s+([\u{4e00}-\u{9fff}\u{3040}-\u{309f}\u{30a0}-\u{30ff}\u{ac00}-\u{d7af}])").unwrap();

        let letter_pattern = [
            r"A-Za-z",
            r"\u{4e00}-\u{9fff}",
            r"\u{3400}-\u{4dbf}",
            r"\u{3040}-\u{309f}",
            r"\u{30a0}-\u{30ff}",
            r"\u{ac00}-\u{d7af}",
            r"\u{1100}-\u{11ff}",
            r"\u{a960}-\u{a97f}",
            r"\u{0400}-\u{04ff}",
            r"\u{0370}-\u{03ff}",
            r"\u{0600}-\u{06ff}",
            r"\u{0590}-\u{05ff}",
            r"\u{0900}-\u{097f}",
            r"\u{0e00}-\u{0e7f}",
            r"\u{1000}-\u{109f}",
            r"\u{10a0}-\u{10ff}",
            r"\u{0530}-\u{058f}",
            r"0-9",
        ]
        .join("");

        let code_block_junk_regex = Regex::new(&format!(r"^[^{}]*", letter_pattern)).unwrap();

        Self {
            html_pre_regex,
            code_fence_regex,
            zero_width_regex,
            unicode_space_regex,
            multiple_newline_regex,
            cjk_latin_space_regex,
            latin_cjk_space_regex,
            code_block_junk_regex,
            cjk_mode,
        }
    }

    fn is_cjk_char(&self, ch: char) -> bool {
        matches!(
            ch.script(),
            Script::Han | Script::Hiragana | Script::Katakana | Script::Hangul
        )
    }

    fn nfkc_normalize(&self, text: &str) -> String {
        // Step 1: Unicode NFKC normalization
        let normalized: String = text.nfkc().collect();

        // Step 2: Normalize line endings
        let unified = normalized.replace("\r\n", "\n").replace('\r', "\n");

        // Step 3: Remove zero-width characters
        let no_zw = self.zero_width_regex.replace_all(&unified, "");

        // Step 4: Convert unicode spaces to regular space (but don't collapse)
        let normalized_spaces = self.unicode_space_regex.replace_all(&no_zw, " ");

        normalized_spaces.to_string()
    }

    fn clean_head_junk(&self, text: &str) -> String {
        let letter_scripts = [
            Script::Latin,
            Script::Han,
            Script::Hiragana,
            Script::Katakana,
            Script::Hangul,
            Script::Cyrillic,
            Script::Greek,
            Script::Arabic,
            Script::Hebrew,
            Script::Devanagari,
            Script::Thai,
            Script::Myanmar,
            Script::Georgian,
            Script::Armenian,
        ];

        for (i, ch) in text.char_indices() {
            if ch.is_ascii_alphanumeric()
                || letter_scripts.contains(&ch.script())
                || (ch as u32 >= 0x1F600 && ch as u32 <= 0x1F64F)
            // emoji
            {
                return text[i..].to_string();
            }
        }

        text.to_string()
    }

    fn pre_clean_code_blocks(&self, text: &str) -> String {
        let junk_regex = &self.code_block_junk_regex;

        self.code_fence_regex
            .replace_all(text, |caps: &regex::Captures| {
                let full_match = caps.get(0).unwrap().as_str();

                // Parse the code block structure
                if let Some(first_newline) = full_match.find('\n') {
                    let prefix = &full_match[..=first_newline];

                    let rest = &full_match[first_newline + 1..];

                    if let Some(last_backticks) = rest.rfind("```") {
                        let content = &rest[..last_backticks];

                        let suffix = &rest[last_backticks..];

                        // Clean the content
                        let cleaned_content = junk_regex.replace(content, "");

                        // If content is too short after cleaning, remove the block
                        if cleaned_content.trim().len() < 10 {
                            return "".to_string();
                        }

                        return format!("{}{}{}", prefix, cleaned_content, suffix);
                    }
                }

                full_match.to_string()
            })
            .to_string()
    }

    fn find_protected_spans(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();

        // Protect HTML pre/code tags
        for m in self.html_pre_regex.find_iter(text) {
            spans.push((m.start(), m.end()));
        }

        // Protect VALID code fences (must contain Latin letters)
        for m in self.code_fence_regex.find_iter(text) {
            let content = m.as_str();

            if content.chars().any(|c| c.is_ascii_alphabetic()) {
                spans.push((m.start(), m.end()));
            }
        }

        // Protect table lines and indent blocks
        let lines: Vec<&str> = text.lines().collect();

        let mut protected_lines = std::collections::HashSet::new();

        // Mark table lines
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with('|')
                && trimmed.ends_with('|')
                && trimmed.matches('|').count() >= 2
            {
                protected_lines.insert(i);
            }
        }

        // Mark consecutive indent blocks (3+ lines)
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

        // Convert line numbers to byte positions
        if !protected_lines.is_empty() {
            let mut pos = 0;

            for (i, line) in lines.iter().enumerate() {
                if protected_lines.contains(&i) {
                    spans.push((pos, pos + line.len()));
                }

                pos += line.len() + 1; // +1 for '\n'
            }
        }

        // Sort and merge overlapping spans
        spans.sort_by(|a, b| a.0.cmp(&b.0));

        let mut merged: Vec<(usize, usize)> = Vec::new();

        for (start, end) in spans {
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

        merged
    }

    fn normalize_line(&self, line: &str) -> String {
        let cleaned = line.trim();

        if cleaned.is_empty() {
            return String::new();
        }

        let mut result = String::with_capacity(cleaned.len());

        let mut chars = cleaned.chars().peekable();

        while let Some(ch) = chars.next() {
            match ch {
                ' ' | '\t' => {
                    // Collapse multiple whitespace
                    while chars.peek() == Some(&' ') || chars.peek() == Some(&'\t') {
                        chars.next();
                    }

                    result.push(' ');
                },
                _ => {
                    result.push(ch);
                },
            }
        }

        result
    }

    fn normalize_body_text(&self, text: &str) -> String {
        // Clean head junk first
        let cleaned = self.clean_head_junk(text);

        // Normalize each line
        let normalized: String = cleaned
            .lines()
            .map(|line| self.normalize_line(line))
            .collect::<Vec<_>>()
            .join("\n");

        let mut result = normalized;

        // CJK processing
        if self.cjk_mode {
            // Remove spaces between CJK characters
            let chars: Vec<char> = result.chars().collect();

            let mut output = String::new();

            let mut i = 0;

            while i < chars.len() {
                let current = chars[i];

                if current == ' ' && i > 0 && i < chars.len() - 1 {
                    let prev = chars[i - 1];

                    let next = chars[i + 1];

                    // Skip space between CJK characters
                    if self.is_cjk_char(prev) && self.is_cjk_char(next) {
                        i += 1;

                        continue;
                    }
                }

                output.push(current);

                i += 1;
            }

            result = output;

            // Ensure proper spacing between CJK and Latin/digits
            result = self
                .cjk_latin_space_regex
                .replace_all(&result, "$1 $2")
                .to_string();

            result = self
                .latin_cjk_space_regex
                .replace_all(&result, "$1 $2")
                .to_string();
        }

        // Collapse multiple newlines
        let final_result = self.multiple_newline_regex.replace_all(&result, "\n\n");

        final_result.trim().to_string()
    }

    fn normalize_text(&self, text: &str) -> (String, usize) {
        if text.trim().is_empty() {
            return (String::new(), 0);
        }

        let original_len = text.len();

        // Step 1: Unicode normalization
        let normalized = self.nfkc_normalize(text);

        // Step 2: Pre-clean code blocks
        let pre_cleaned = self.pre_clean_code_blocks(&normalized);

        // Step 3: Find protected regions
        let protected_spans = self.find_protected_spans(&pre_cleaned);

        let result = if protected_spans.is_empty() {
            // No protected regions, process all text
            self.normalize_body_text(&pre_cleaned)
        } else {
            // Process in segments
            let mut pieces = Vec::new();

            let mut last_pos = 0;

            for (start, end) in protected_spans {
                // Process unprotected region before this span
                if start > last_pos {
                    let body_part = &pre_cleaned[last_pos..start];

                    pieces.push(self.normalize_body_text(body_part));
                }

                // Keep protected region as-is
                pieces.push(pre_cleaned[start..end].to_string());

                last_pos = end;
            }

            // Process final unprotected region
            if last_pos < pre_cleaned.len() {
                let body_part = &pre_cleaned[last_pos..];

                pieces.push(self.normalize_body_text(body_part));
            }

            pieces.join("")
        };

        let chars_removed = original_len.saturating_sub(result.len());

        (result, chars_removed)
    }
}

pub struct TextNormalizationFormatter {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
    normalizer: TextNormalizer,
    min_keep_length: usize,
}

impl TextNormalizationFormatter {
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self {
            writer,
            normalizer: TextNormalizer::new(true),
            min_keep_length: 50,
        }
    }
}

impl Step for TextNormalizationFormatter {
    fn name(&self) -> &'static str {
        "TextNormalization"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        let normalizer = &self.normalizer;

        let min_keep_length = self.min_keep_length;

        batch
            .into_par_iter()
            .map(|data| {
                let (normalized, _chars_removed) = normalizer.normalize_text(data.as_ref());

                if normalized.len() < min_keep_length {
                    StepOutcome::Exclude(data)
                } else {
                    StepOutcome::Keep(Cow::Owned(normalized))
                }
            })
            .collect()
    }
}
