use crate::inference_core::{InferenceOutput, StreamDelta};

#[derive(Debug, Default)]
pub struct ByteDecoder {
    stop_suffixes: Vec<Vec<u8>>,
    max_stop_suffix_len: usize,
    generated_bytes: Vec<u8>,
    generated_tokens: Vec<InferenceOutput>,
    emitted_token_count: usize,
    emitted_byte_len: usize,
}

impl ByteDecoder {
    pub fn new(stop_suffixes: Vec<String>) -> Self {
        let stop_suffixes: Vec<Vec<u8>> = stop_suffixes
            .into_iter()
            .filter(|suffix| !suffix.is_empty())
            .map(String::into_bytes)
            .collect();
        let max_stop_suffix_len = stop_suffixes
            .iter()
            .map(|suffix| suffix.len())
            .max()
            .unwrap_or(0);
        Self {
            stop_suffixes,
            max_stop_suffix_len,
            generated_bytes: Vec::new(),
            generated_tokens: Vec::new(),
            emitted_token_count: 0,
            emitted_byte_len: 0,
        }
    }

    pub fn push_output(&mut self, output: InferenceOutput) -> StreamDelta {
        self.generated_bytes.extend_from_slice(&output.bytes);
        self.generated_tokens.push(output);
        let hold_len = self.max_stop_suffix_len.saturating_sub(1);
        let emit_limit = self.generated_bytes.len().saturating_sub(hold_len);
        self.flush_stream_delta_until(emit_limit, false)
    }

    pub fn finish(&mut self, matched_stop_suffix_index: Option<usize>) -> StreamDelta {
        let emit_limit = matched_stop_suffix_index
            .and_then(|index| self.stop_suffixes.get(index))
            .map(|suffix| self.generated_bytes.len().saturating_sub(suffix.len()))
            .unwrap_or(self.generated_bytes.len());
        self.flush_stream_delta_until(emit_limit, true)
    }

    fn flush_stream_delta_until(
        &mut self,
        emit_limit: usize,
        allow_partial_token: bool,
    ) -> StreamDelta {
        if self.emitted_byte_len >= emit_limit {
            return StreamDelta::default();
        }

        let mut full_token_limit = self.emitted_token_count;
        let mut byte_cursor = self.emitted_byte_len;
        while let Some(token) = self.generated_tokens.get(full_token_limit) {
            let next_byte_cursor = byte_cursor + token.bytes.len();
            if next_byte_cursor > emit_limit {
                break;
            }
            byte_cursor = next_byte_cursor;
            full_token_limit += 1;
        }

        let candidate_tokens = &self.generated_tokens[self.emitted_token_count..full_token_limit];
        let emit_full_count = longest_valid_utf8_token_prefix(candidate_tokens);
        let emit_tokens_end = self.emitted_token_count + emit_full_count;

        let mut emitted_tokens =
            self.generated_tokens[self.emitted_token_count..emit_tokens_end].to_vec();
        let mut emitted_text = decode_tokens_text(&emitted_tokens);

        self.emitted_token_count = emit_tokens_end;
        self.emitted_byte_len += emitted_tokens
            .iter()
            .map(|token| token.bytes.len())
            .sum::<usize>();

        if allow_partial_token && self.emitted_byte_len < emit_limit {
            let partial_len = emit_limit - self.emitted_byte_len;
            if let Some(token) = self.generated_tokens.get(self.emitted_token_count) {
                let partial_bytes = &token.bytes[..partial_len.min(token.bytes.len())];
                let valid_prefix_len = longest_valid_utf8_prefix_len(partial_bytes);
                if valid_prefix_len > 0 {
                    let bytes = partial_bytes[..valid_prefix_len].to_vec();
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    emitted_text.push_str(&text);
                    emitted_tokens.push(InferenceOutput {
                        token: text,
                        bytes,
                        logprob: token.logprob,
                        top_logprobs: token.top_logprobs.clone(),
                    });
                    self.emitted_byte_len += valid_prefix_len;
                }
            }
        }

        StreamDelta {
            text: emitted_text,
            tokens: emitted_tokens,
        }
    }
}

fn longest_valid_utf8_token_prefix(tokens: &[InferenceOutput]) -> usize {
    if tokens.is_empty() {
        return 0;
    }

    let mut buf = Vec::new();
    let mut last_valid = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        buf.extend_from_slice(&token.bytes);
        match std::str::from_utf8(&buf) {
            Ok(_) => last_valid = index + 1,
            Err(err) if err.error_len().is_none() => {}
            Err(_) => {
                last_valid = index + 1;
                break;
            }
        }
    }
    last_valid
}

fn decode_tokens_text(tokens: &[InferenceOutput]) -> String {
    if tokens.is_empty() {
        return String::new();
    }

    let mut bytes = Vec::new();
    for token in tokens {
        bytes.extend_from_slice(&token.bytes);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn longest_valid_utf8_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(err) => err.valid_up_to(),
    }
}

#[cfg(test)]
mod tests {
    use super::ByteDecoder;
    use crate::inference_core::{InferenceOutput, InferenceOutputCandidate};

    #[test]
    fn flush_stream_delta_emits_partial_visible_prefix_on_stop() {
        let mut decoder = ByteDecoder::new(vec!["END".to_string()]);
        let delta = decoder.push_output(InferenceOutput {
            token: " THE END".to_string(),
            bytes: b" THE END".to_vec(),
            logprob: Some(-0.5),
            top_logprobs: vec![InferenceOutputCandidate {
                token: " THE END".to_string(),
                bytes: b" THE END".to_vec(),
                logprob: -0.5,
            }],
        });
        assert!(delta.text.is_empty());

        let delta = decoder.finish(Some(0));
        assert_eq!(delta.text, " THE ");
        assert_eq!(delta.tokens.len(), 1);
        assert_eq!(delta.tokens[0].token, " THE ");
        assert_eq!(delta.tokens[0].bytes, b" THE ".to_vec());
    }

    #[test]
    fn preserves_utf8_boundaries_until_finish() {
        let mut decoder = ByteDecoder::new(Vec::new());
        let delta = decoder.push_output(InferenceOutput {
            token: "你".to_string(),
            bytes: vec![0xE4, 0xBD, 0xA0],
            logprob: None,
            top_logprobs: Vec::new(),
        });
        assert_eq!(delta.text, "你");
        assert_eq!(delta.tokens.len(), 1);
    }
}
