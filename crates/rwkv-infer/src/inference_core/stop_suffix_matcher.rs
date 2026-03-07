#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StopMatch {
    pub index: usize,
    pub len: usize,
}

#[derive(Debug, Default)]
pub struct StopSuffixMatcher {
    suffixes: Vec<Vec<u8>>,
    max_suffix_len: usize,
    trailing_bytes: Vec<u8>,
}

impl StopSuffixMatcher {
    pub fn new(stop_suffixes: Vec<String>) -> Self {
        let suffixes: Vec<Vec<u8>> = stop_suffixes
            .into_iter()
            .filter(|suffix| !suffix.is_empty())
            .map(String::into_bytes)
            .collect();
        let max_suffix_len = suffixes
            .iter()
            .map(|suffix| suffix.len())
            .max()
            .unwrap_or(0);
        Self {
            suffixes,
            max_suffix_len,
            trailing_bytes: Vec::new(),
        }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Option<StopMatch> {
        if self.max_suffix_len == 0 || bytes.is_empty() {
            return None;
        }

        self.trailing_bytes.extend_from_slice(bytes);
        if self.trailing_bytes.len() > self.max_suffix_len {
            let keep_from = self.trailing_bytes.len() - self.max_suffix_len;
            self.trailing_bytes.drain(..keep_from);
        }

        match_stop_suffix(&self.trailing_bytes, &self.suffixes)
    }

    pub fn suffix(&self, index: usize) -> Option<&[u8]> {
        self.suffixes.get(index).map(Vec::as_slice)
    }

    pub fn max_suffix_len(&self) -> usize {
        self.max_suffix_len
    }
}

fn match_stop_suffix(output: &[u8], suffixes: &[Vec<u8>]) -> Option<StopMatch> {
    let mut best = None;
    for (index, suffix) in suffixes.iter().enumerate() {
        if suffix.is_empty() || !output.ends_with(suffix.as_slice()) {
            continue;
        }
        let candidate = StopMatch {
            index,
            len: suffix.len(),
        };
        match best {
            None => best = Some(candidate),
            Some(current) => {
                if candidate.len > current.len
                    || (candidate.len == current.len && candidate.index < current.index)
                {
                    best = Some(candidate);
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{StopSuffixMatcher, match_stop_suffix};

    #[test]
    fn match_stop_suffix_prefers_longer_suffix() {
        let suffixes = vec![b"END".to_vec(), b"THE END".to_vec()];
        let output = b"This is THE END";
        let matched = match_stop_suffix(output, &suffixes).expect("must match");
        assert_eq!(matched.index, 1);
        assert_eq!(matched.len, 7);
    }

    #[test]
    fn match_stop_suffix_prefers_lower_index_on_tie() {
        let suffixes = vec![b"stop".to_vec(), b"stop".to_vec()];
        let output = b"please stop";
        let matched = match_stop_suffix(output, &suffixes).expect("must match");
        assert_eq!(matched.index, 0);
        assert_eq!(matched.len, 4);
    }

    #[test]
    fn matcher_detects_suffix_across_token_boundaries() {
        let mut matcher = StopSuffixMatcher::new(vec!["END".to_string()]);
        assert_eq!(matcher.push_bytes(b"E"), None);
        assert_eq!(matcher.push_bytes(b"N"), None);
        let matched = matcher.push_bytes(b"D").expect("must match");
        assert_eq!(matched.index, 0);
        assert_eq!(matched.len, 3);
    }
}
