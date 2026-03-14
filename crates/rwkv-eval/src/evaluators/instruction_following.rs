use crate::datasets::SamplingConfig;
use crate::inferers::{CompletionRequest, CompletionResponse};
use async_openai::Client;
use async_openai::config::OpenAIConfig;
use langdetect_rs::detector_factory::DetectorFactory;
use regex::{Captures, Regex, RegexBuilder};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionKind {
    CapitalWordFrequency,
    EnglishCapital,
    EnglishLowercase,
    RepeatPrompt,
    TwoResponses,
    NumberPlaceholders,
    Postscript,
    ConstrainedResponse,
    JsonFormat,
    MultipleSections,
    NumberBulletLists,
    NumberHighlightedSections,
    Title,
    KeywordExistence,
    KeywordForbiddenWords,
    KeywordFrequency,
    LetterFrequency,
    ResponseLanguage,
    NthParagraphFirstWord,
    NumberParagraphs,
    NumberSentences,
    NumberWords,
    NoComma,
    EndChecker,
    Quotation,
}

#[derive(Debug, Clone)]
pub struct InstructionSpec {
    pub kind: InstructionKind,
    pub args: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct InstructionCheck {
    pub kind: InstructionKind,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct InstructionFollowingEvalResult {
    pub all_passed: bool,
    pub checks: Vec<InstructionCheck>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareRelation {
    LessThan,
    AtLeast,
}

impl InstructionKind {
    fn parse(id: &str) -> Self {
        match id {
            "change_case:capital_word_frequency" => Self::CapitalWordFrequency,
            "change_case:english_capital" => Self::EnglishCapital,
            "change_case:english_lowercase" => Self::EnglishLowercase,
            "combination:repeat_prompt" => Self::RepeatPrompt,
            "combination:two_responses" => Self::TwoResponses,
            "detectable_content:number_placeholders" => Self::NumberPlaceholders,
            "detectable_content:postscript" => Self::Postscript,
            "detectable_format:constrained_response" => Self::ConstrainedResponse,
            "detectable_format:json_format" => Self::JsonFormat,
            "detectable_format:multiple_sections" => Self::MultipleSections,
            "detectable_format:number_bullet_lists" => Self::NumberBulletLists,
            "detectable_format:number_highlighted_sections" => Self::NumberHighlightedSections,
            "detectable_format:title" => Self::Title,
            "keywords:existence" => Self::KeywordExistence,
            "keywords:forbidden_words" => Self::KeywordForbiddenWords,
            "keywords:frequency" => Self::KeywordFrequency,
            "keywords:letter_frequency" => Self::LetterFrequency,
            "language:response_language" => Self::ResponseLanguage,
            "length_constraints:nth_paragraph_first_word" => Self::NthParagraphFirstWord,
            "length_constraints:number_paragraphs" => Self::NumberParagraphs,
            "length_constraints:number_sentences" => Self::NumberSentences,
            "length_constraints:number_words" => Self::NumberWords,
            "punctuation:no_comma" => Self::NoComma,
            "startend:end_checker" => Self::EndChecker,
            "startend:quotation" => Self::Quotation,
            _ => panic!("unsupported instruction id: {id}"),
        }
    }

    fn raw_id(self) -> &'static str {
        match self {
            Self::CapitalWordFrequency => "change_case:capital_word_frequency",
            Self::EnglishCapital => "change_case:english_capital",
            Self::EnglishLowercase => "change_case:english_lowercase",
            Self::RepeatPrompt => "combination:repeat_prompt",
            Self::TwoResponses => "combination:two_responses",
            Self::NumberPlaceholders => "detectable_content:number_placeholders",
            Self::Postscript => "detectable_content:postscript",
            Self::ConstrainedResponse => "detectable_format:constrained_response",
            Self::JsonFormat => "detectable_format:json_format",
            Self::MultipleSections => "detectable_format:multiple_sections",
            Self::NumberBulletLists => "detectable_format:number_bullet_lists",
            Self::NumberHighlightedSections => "detectable_format:number_highlighted_sections",
            Self::Title => "detectable_format:title",
            Self::KeywordExistence => "keywords:existence",
            Self::KeywordForbiddenWords => "keywords:forbidden_words",
            Self::KeywordFrequency => "keywords:frequency",
            Self::LetterFrequency => "keywords:letter_frequency",
            Self::ResponseLanguage => "language:response_language",
            Self::NthParagraphFirstWord => "length_constraints:nth_paragraph_first_word",
            Self::NumberParagraphs => "length_constraints:number_paragraphs",
            Self::NumberSentences => "length_constraints:number_sentences",
            Self::NumberWords => "length_constraints:number_words",
            Self::NoComma => "punctuation:no_comma",
            Self::EndChecker => "startend:end_checker",
            Self::Quotation => "startend:quotation",
        }
    }
}

impl InstructionSpec {
    pub fn new(id: impl AsRef<str>, args: Map<String, Value>) -> Self {
        Self {
            kind: InstructionKind::parse(id.as_ref()),
            args,
        }
    }
}

impl CompareRelation {
    fn parse(value: &str) -> Self {
        match value {
            "less than" => Self::LessThan,
            "at least" => Self::AtLeast,
            _ => panic!("unsupported comparison relation: {value}"),
        }
    }

    fn matches(self, actual: usize, threshold: usize) -> bool {
        match self {
            Self::LessThan => actual < threshold,
            Self::AtLeast => actual >= threshold,
        }
    }
}

pub fn build_prompt(prompt: &str) -> String {
    format!("User: {prompt}\n\nAssistant:")
}

pub async fn generate_response(
    model_client: &Client<OpenAIConfig>,
    model_name: &str,
    prompt: &str,
    sampling_config: &SamplingConfig,
) -> String {
    let req = CompletionRequest::new(
        model_name.to_string(),
        prompt.to_string().into(),
        vec![],
        4096,
        sampling_config,
        None,
        None,
    );

    let resp: CompletionResponse = model_client.completions().create_byot(&req).await.unwrap();
    resp.choices[0].text.clone()
}

pub fn describe_instructions(instructions: &[InstructionSpec]) -> String {
    instructions
        .iter()
        .map(|instruction| {
            if instruction.args.is_empty() {
                instruction.kind.raw_id().to_string()
            } else {
                format!(
                    "{} {}",
                    instruction.kind.raw_id(),
                    serde_json::to_string(&instruction.args).unwrap()
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn evaluate_response(
    instructions: &[InstructionSpec],
    response: &str,
) -> InstructionFollowingEvalResult {
    let checks = instructions
        .iter()
        .map(|instruction| InstructionCheck {
            kind: instruction.kind,
            passed: check_instruction(instruction, response),
        })
        .collect::<Vec<_>>();

    InstructionFollowingEvalResult {
        all_passed: !checks.is_empty() && checks.iter().all(|check| check.passed),
        checks,
    }
}

fn check_instruction(instruction: &InstructionSpec, response: &str) -> bool {
    match instruction.kind {
        InstructionKind::CapitalWordFrequency => {
            let relation = CompareRelation::parse(arg_str(instruction, "capital_relation"));
            relation.matches(
                count_capital_words(response),
                arg_usize(instruction, "capital_frequency"),
            )
        }
        InstructionKind::EnglishCapital => {
            is_all_uppercase(response) && detect_language_matches("en", response)
        }
        InstructionKind::EnglishLowercase => {
            is_all_lowercase(response) && detect_language_matches("en", response)
        }
        InstructionKind::RepeatPrompt => response.trim().to_lowercase().starts_with(
            &arg_str(instruction, "prompt_to_repeat")
                .trim()
                .to_lowercase(),
        ),
        InstructionKind::TwoResponses => check_two_responses(response),
        InstructionKind::NumberPlaceholders => {
            Regex::new(r"\[.*?\]").unwrap().find_iter(response).count()
                >= arg_usize(instruction, "num_placeholders")
        }
        InstructionKind::Postscript => {
            check_postscript(response, arg_str(instruction, "postscript_marker"))
        }
        InstructionKind::ConstrainedResponse => {
            let value = response.trim();
            [
                "My answer is yes.",
                "My answer is no.",
                "My answer is maybe.",
            ]
            .iter()
            .any(|option| value.contains(option))
        }
        InstructionKind::JsonFormat => check_json_format(response),
        InstructionKind::MultipleSections => {
            count_sections(response, arg_str(instruction, "section_spliter"))
                >= arg_usize(instruction, "num_sections")
        }
        InstructionKind::NumberBulletLists => {
            count_bullet_lists(response) == arg_usize(instruction, "num_bullets")
        }
        InstructionKind::NumberHighlightedSections => {
            count_highlighted_sections(response) >= arg_usize(instruction, "num_highlights")
        }
        InstructionKind::Title => {
            Regex::new(r"<<[^\n]+>>")
                .unwrap()
                .find_iter(response)
                .any(|title| {
                    !title
                        .as_str()
                        .trim_start_matches('<')
                        .trim_end_matches('>')
                        .trim()
                        .is_empty()
                })
        }
        InstructionKind::KeywordExistence => arg_str_list(instruction, "keywords")
            .into_iter()
            .all(|keyword| contains_regex(keyword, response)),
        InstructionKind::KeywordForbiddenWords => arg_str_list(instruction, "forbidden_words")
            .into_iter()
            .all(|word| !contains_whole_word_regex(word, response)),
        InstructionKind::KeywordFrequency => {
            let relation = CompareRelation::parse(arg_str(instruction, "relation"));
            relation.matches(
                count_regex_occurrences(arg_str(instruction, "keyword"), response),
                arg_usize(instruction, "frequency"),
            )
        }
        InstructionKind::LetterFrequency => {
            let relation = CompareRelation::parse(arg_str(instruction, "let_relation"));
            relation.matches(
                response
                    .chars()
                    .filter(|ch| {
                        ch.to_lowercase().to_string()
                            == arg_str(instruction, "letter").to_lowercase()
                    })
                    .count(),
                arg_usize(instruction, "let_frequency"),
            )
        }
        InstructionKind::ResponseLanguage => {
            detect_language_matches(arg_str(instruction, "language"), response)
        }
        InstructionKind::NthParagraphFirstWord => check_nth_paragraph_first_word(
            response,
            arg_usize(instruction, "num_paragraphs"),
            arg_usize(instruction, "nth_paragraph"),
            arg_str(instruction, "first_word"),
        ),
        InstructionKind::NumberParagraphs => {
            check_number_paragraphs(response, arg_usize(instruction, "num_paragraphs"))
        }
        InstructionKind::NumberSentences => {
            let relation = CompareRelation::parse(arg_str(instruction, "relation"));
            relation.matches(
                count_sentences(response),
                arg_usize(instruction, "num_sentences"),
            )
        }
        InstructionKind::NumberWords => {
            let relation = CompareRelation::parse(arg_str(instruction, "relation"));
            relation.matches(count_words(response), arg_usize(instruction, "num_words"))
        }
        InstructionKind::NoComma => !response.contains(','),
        InstructionKind::EndChecker => response
            .trim()
            .trim_matches('"')
            .to_lowercase()
            .ends_with(&arg_str(instruction, "end_phrase").trim().to_lowercase()),
        InstructionKind::Quotation => {
            let value = response.trim();
            value.len() > 1 && value.starts_with('"') && value.ends_with('"')
        }
    }
}

fn arg_str<'a>(instruction: &'a InstructionSpec, key: &str) -> &'a str {
    instruction
        .args
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            panic!(
                "missing string arg `{key}` for {}",
                instruction.kind.raw_id()
            )
        })
}

fn arg_usize(instruction: &InstructionSpec, key: &str) -> usize {
    instruction
        .args
        .get(key)
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            panic!(
                "missing integer arg `{key}` for {}",
                instruction.kind.raw_id()
            )
        }) as usize
}

fn arg_str_list<'a>(instruction: &'a InstructionSpec, key: &str) -> Vec<&'a str> {
    instruction
        .args
        .get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "missing string-list arg `{key}` for {}",
                instruction.kind.raw_id()
            )
        })
        .iter()
        .map(|value| {
            value.as_str().unwrap_or_else(|| {
                panic!(
                    "non-string element in `{key}` for {}",
                    instruction.kind.raw_id()
                )
            })
        })
        .collect()
}

fn contains_regex(pattern: &str, value: &str) -> bool {
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .unwrap()
        .is_match(value)
}

fn contains_whole_word_regex(pattern: &str, value: &str) -> bool {
    RegexBuilder::new(&format!(r"\b{pattern}\b"))
        .case_insensitive(true)
        .build()
        .unwrap()
        .is_match(value)
}

fn count_regex_occurrences(pattern: &str, value: &str) -> usize {
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .unwrap()
        .find_iter(value)
        .count()
}

fn count_words(value: &str) -> usize {
    Regex::new(r"\w+").unwrap().find_iter(value).count()
}

fn count_bullet_lists(value: &str) -> usize {
    Regex::new(r"(?m)^\s*\*[^\*].*$")
        .unwrap()
        .find_iter(value)
        .count()
        + Regex::new(r"(?m)^\s*-.*$")
            .unwrap()
            .find_iter(value)
            .count()
}

fn count_highlighted_sections(value: &str) -> usize {
    let single_count = Regex::new(r"\*[^\n\*]*\*")
        .unwrap()
        .find_iter(value)
        .filter(|matched| !matched.as_str().trim_matches('*').trim().is_empty())
        .count();
    let double_count = Regex::new(r"\*\*[^\n\*]*\*\*")
        .unwrap()
        .find_iter(value)
        .filter(|matched| {
            !matched
                .as_str()
                .trim_start_matches("**")
                .trim_end_matches("**")
                .trim()
                .is_empty()
        })
        .count();

    single_count + (double_count * 2)
}

fn count_capital_words(value: &str) -> usize {
    treebank_word_tokens(value)
        .into_iter()
        .filter(|word| is_all_uppercase(word))
        .count()
}

fn apply_regex_replacements(mut text: String, patterns: &[(&str, &str)]) -> String {
    for (pattern, replacement) in patterns {
        text = Regex::new(pattern)
            .unwrap()
            .replace_all(&text, *replacement)
            .into_owned();
    }
    text
}

fn treebank_word_tokens(value: &str) -> Vec<String> {
    let mut text = apply_regex_replacements(
        value.to_string(),
        &[
            (r#"^""#, "``"),
            (r"(``)", " $1 "),
            (r#"([ \(\[{<])("|'{2})"#, "$1 `` "),
            (r"([:,])([^\d])", " $1 $2"),
            (r"([:,])$", " $1 "),
            (r"\.\.\.", " ... "),
            (r"[;@#$%&]", " $0 "),
            (r#"([^\.])(\.)([\]\)}>"']*)\s*$"#, "$1 $2$3 "),
            (r"[?!]", " $0 "),
            (r"([^'])' ", "$1 ' "),
            (r"[\]\[\(\)\{\}<>]", " $0 "),
            (r"--", " -- "),
        ],
    );

    text.insert(0, ' ');
    text.push(' ');

    apply_regex_replacements(
        text,
        &[
            (r"''", " '' "),
            (r#"""#, " '' "),
            (r"([^' ])('[sS]|'[mM]|'[dD]|') ", "$1 $2 "),
            (r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ", "$1 $2 "),
            (r"(?i)\b(can)(not)\b", " $1 $2 "),
            (r"(?i)\b(d)('ye)\b", " $1 $2 "),
            (r"(?i)\b(gim)(me)\b", " $1 $2 "),
            (r"(?i)\b(gon)(na)\b", " $1 $2 "),
            (r"(?i)\b(got)(ta)\b", " $1 $2 "),
            (r"(?i)\b(lem)(me)\b", " $1 $2 "),
            (r"(?i)\b(more)('n)\b", " $1 $2 "),
            (r"(?i)\b(wan)(na)(\s)", " $1 $2$3"),
            (r"(?i) ('t)(is)\b", " $1 $2 "),
            (r"(?i) ('t)(was)\b", " $1 $2 "),
        ],
    )
    .split_whitespace()
    .map(str::to_owned)
    .collect()
}

fn check_two_responses(value: &str) -> bool {
    let responses = value.split("******").collect::<Vec<_>>();
    let mut valid = Vec::new();

    for (index, response) in responses.iter().enumerate() {
        if response.trim().is_empty() {
            if index != 0 && index + 1 != responses.len() {
                return false;
            }
            continue;
        }
        valid.push(response.trim());
    }

    valid.len() == 2 && valid[0] != valid[1]
}

fn check_postscript(value: &str, marker: &str) -> bool {
    let value = value.to_lowercase();
    let pattern = match marker {
        "P.P.S" => r"(?m)\s*p\.\s?p\.\s?s.*$".to_string(),
        "P.S." => r"(?m)\s*p\.\s?s\..*$".to_string(),
        _ => format!(r"(?m)\s*{}.*$", regex::escape(&marker.to_lowercase())),
    };

    Regex::new(&pattern).unwrap().is_match(&value)
}

fn check_json_format(value: &str) -> bool {
    let value = value.trim();
    let value = value
        .strip_prefix("```json")
        .or_else(|| value.strip_prefix("```Json"))
        .or_else(|| value.strip_prefix("```JSON"))
        .or_else(|| value.strip_prefix("```"))
        .unwrap_or(value)
        .trim();
    let value = value.strip_suffix("```").unwrap_or(value).trim();
    serde_json::from_str::<Value>(value).is_ok()
}

fn count_sections(value: &str, splitter: &str) -> usize {
    Regex::new(&format!(r"\s?{}\s?\d+\s?", regex::escape(splitter)))
        .unwrap()
        .split(value)
        .count()
        .saturating_sub(1)
}

fn check_number_paragraphs(value: &str, expected: usize) -> bool {
    let paragraphs = Regex::new(r"\s?\*\*\*\s?")
        .unwrap()
        .split(value)
        .collect::<Vec<_>>();
    let mut count = paragraphs.len();

    for (index, paragraph) in paragraphs.iter().enumerate() {
        if paragraph.trim().is_empty() {
            if index == 0 || index + 1 == paragraphs.len() {
                count -= 1;
            } else {
                return false;
            }
        }
    }

    count == expected
}

fn check_nth_paragraph_first_word(
    value: &str,
    expected_count: usize,
    nth_paragraph: usize,
    first_word: &str,
) -> bool {
    let paragraphs = value.split("\n\n").collect::<Vec<_>>();
    let mut count = paragraphs.len();
    for paragraph in &paragraphs {
        if paragraph.trim().is_empty() {
            count -= 1;
        }
    }

    if count != expected_count || nth_paragraph == 0 || nth_paragraph > count {
        return false;
    }

    let paragraph = paragraphs[nth_paragraph - 1].trim();
    if paragraph.is_empty() {
        return false;
    }

    let mut actual = String::new();
    let mut word = paragraph.split_whitespace().next().unwrap().trim();
    word = word.trim_start_matches('\'');
    word = word.trim_start_matches('"');

    for ch in word.chars() {
        if matches!(ch, '.' | ',' | '?' | '!' | '\'' | '"') {
            break;
        }
        actual.extend(ch.to_lowercase());
    }

    actual == first_word.to_lowercase()
}

fn count_sentences(value: &str) -> usize {
    let mut text = format!(" {value}  ").replace('\n', " ");
    text = text
        .replace("P.P.S.", "P<prd>P<prd>S<ps_end>")
        .replace("P.S.", "P<prd>S<ps_end>");
    text = apply_regex_replacements(
        text,
        &[
            (r"(Mr|St|Mrs|Ms|Dr)[.]", "$1<prd>"),
            (r"[.](com|net|org|io|gov|edu|me)", "<prd>$1"),
            (r"([0-9])[.]([0-9])", "$1<prd>$2"),
        ],
    );
    text = Regex::new(r"\.{2,}")
        .unwrap()
        .replace_all(&text, |matched: &Captures| {
            "<prd>".repeat(matched[0].chars().count())
        })
        .into_owned();
    if text.contains("Ph.D") {
        text = text.replace("Ph.D.", "Ph<prd>D<prd>");
    }
    text = apply_regex_replacements(
        text,
        &[
            (r"\s([A-Za-z])[.] ", " $1<prd> "),
            (
                r"([A-Z][.][A-Z][.](?:[A-Z][.])?) (Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)",
                "$1<stop> $2",
            ),
            (
                r"([A-Za-z])[.]([A-Za-z])[.]([A-Za-z])[.]",
                "$1<prd>$2<prd>$3<prd>",
            ),
            (r"([A-Za-z])[.]([A-Za-z])[.]", "$1<prd>$2<prd>"),
            (
                r" (Inc|Ltd|Jr|Sr|Co)[.] (Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)",
                " $1<stop> $2",
            ),
            (r" (Inc|Ltd|Jr|Sr|Co)[.]", " $1<prd>"),
            (r" ([A-Za-z])[.]", " $1<prd>"),
        ],
    );
    if text.contains('”') {
        text = text.replace(".”", "”.");
    }
    if text.contains('"') {
        text = text.replace(".\"", "\".");
    }
    if text.contains('!') {
        text = text.replace("!\"", "\"!");
    }
    if text.contains('?') {
        text = text.replace("?\"", "\"?");
    }
    text = text
        .replace('.', ".<stop>")
        .replace('?', "?<stop>")
        .replace('!', "!<stop>")
        .replace("<ps_end>", ".<stop>")
        .replace("<prd>", ".");
    let mut sentences = text.split("<stop>").map(str::trim).collect::<Vec<_>>();
    if matches!(sentences.last(), Some(last) if last.is_empty()) {
        sentences.pop();
    }
    sentences
        .into_iter()
        .filter(|sentence| !sentence.is_empty())
        .count()
}

fn is_all_uppercase(value: &str) -> bool {
    let mut has_cased = false;
    for ch in value.chars() {
        if ch.is_lowercase() {
            return false;
        }
        if ch.is_uppercase() {
            has_cased = true;
        }
    }
    has_cased
}

fn is_all_lowercase(value: &str) -> bool {
    let mut has_cased = false;
    for ch in value.chars() {
        if ch.is_uppercase() {
            return false;
        }
        if ch.is_lowercase() {
            has_cased = true;
        }
    }
    has_cased
}

fn detect_language_matches(language: &str, value: &str) -> bool {
    DetectorFactory::default()
        .build()
        .detect(value, None)
        .map(|detected| detected == language)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::{
        InstructionSpec, build_prompt, check_json_format, count_capital_words,
        count_highlighted_sections, count_sentences, evaluate_response,
    };
    use serde_json::{Map, Value, json};

    fn spec(id: &str, args: Value) -> InstructionSpec {
        let Value::Object(args) = args else {
            panic!("expected object args");
        };
        InstructionSpec::new(id, args)
    }

    #[test]
    fn parses_all_ifeval_instruction_ids() {
        let ids = [
            "change_case:capital_word_frequency",
            "change_case:english_capital",
            "change_case:english_lowercase",
            "combination:repeat_prompt",
            "combination:two_responses",
            "detectable_content:number_placeholders",
            "detectable_content:postscript",
            "detectable_format:constrained_response",
            "detectable_format:json_format",
            "detectable_format:multiple_sections",
            "detectable_format:number_bullet_lists",
            "detectable_format:number_highlighted_sections",
            "detectable_format:title",
            "keywords:existence",
            "keywords:forbidden_words",
            "keywords:frequency",
            "keywords:letter_frequency",
            "language:response_language",
            "length_constraints:nth_paragraph_first_word",
            "length_constraints:number_paragraphs",
            "length_constraints:number_sentences",
            "length_constraints:number_words",
            "punctuation:no_comma",
            "startend:end_checker",
            "startend:quotation",
        ];

        for id in ids {
            let _ = InstructionSpec::new(id, Map::new());
        }
    }

    #[test]
    fn builds_expected_prompt() {
        assert_eq!(build_prompt("Hello"), "User: Hello\n\nAssistant:");
    }

    #[test]
    fn checks_format_constraints() {
        assert!(
            evaluate_response(
                &[
                    spec(
                        "detectable_format:number_bullet_lists",
                        json!({"num_bullets": 2})
                    ),
                    spec("detectable_format:title", json!({})),
                ],
                "<<Plan>>\n* first\n- second"
            )
            .all_passed
        );

        assert!(
            !evaluate_response(
                &[spec(
                    "detectable_format:number_bullet_lists",
                    json!({"num_bullets": 2})
                )],
                "* first\n* second\n* third"
            )
            .all_passed
        );
    }

    #[test]
    fn checks_keyword_and_length_constraints() {
        assert!(
            evaluate_response(
                &[
                    spec(
                        "keywords:existence",
                        json!({"keywords": ["farmer", "apple"]})
                    ),
                    spec(
                        "length_constraints:number_words",
                        json!({"num_words": 6, "relation": "at least"})
                    ),
                ],
                "A farmer picked one apple near another apple tree."
            )
            .all_passed
        );

        assert!(
            !evaluate_response(
                &[spec(
                    "keywords:forbidden_words",
                    json!({"forbidden_words": ["economy"]})
                )],
                "The economy is improving."
            )
            .all_passed
        );
    }

    #[test]
    fn checks_combination_and_end_constraints() {
        assert!(
            evaluate_response(
                &[
                    spec(
                        "combination:repeat_prompt",
                        json!({"prompt_to_repeat": "Say hello"})
                    ),
                    spec("startend:end_checker", json!({"end_phrase": "goodbye"})),
                ],
                "Say hello and then answer goodbye"
            )
            .checks[0]
                .passed
        );

        assert!(
            evaluate_response(
                &[
                    spec("combination:two_responses", json!({})),
                    spec("startend:quotation", json!({})),
                ],
                "\"first\"******\"second\""
            )
            .checks
            .iter()
            .all(|check| check.passed)
        );
    }

    #[test]
    fn checks_paragraph_and_sentence_constraints() {
        assert!(
            evaluate_response(
                &[spec(
                    "length_constraints:number_paragraphs",
                    json!({"num_paragraphs": 2})
                )],
                "First paragraph.\n***\nSecond paragraph."
            )
            .all_passed
        );

        assert!(
            evaluate_response(
                &[
                    spec(
                        "length_constraints:nth_paragraph_first_word",
                        json!({"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "alpha"})
                    ),
                    spec(
                        "length_constraints:number_sentences",
                        json!({"num_sentences": 2, "relation": "at least"})
                    ),
                ],
                "First paragraph.\n\nAlpha starts here. Another sentence."
            )
            .checks
            .iter()
            .all(|check| check.passed)
        );
    }

    #[test]
    fn checks_json_and_highlight_behavior() {
        assert!(check_json_format("```json\n{\"a\":1}\n```"));
        assert_eq!(count_highlighted_sections("**bold**"), 2);
    }

    #[test]
    fn checks_case_constraints() {
        assert!(
            evaluate_response(
                &[spec("change_case:english_capital", json!({}))],
                "THIS ENTIRE RESPONSE IS WRITTEN ONLY IN ENGLISH."
            )
            .all_passed
        );
        assert!(
            evaluate_response(
                &[spec("change_case:english_lowercase", json!({}))],
                "this entire response is written only in english."
            )
            .all_passed
        );
    }

    #[test]
    fn counts_capital_words_like_treebank_tokenizer_samples() {
        let cases = [
            ("HELLO-WORLD", 1),
            ("U.S.A", 1),
            ("U.S.A.", 1),
            ("ABC123", 1),
            ("HELLO, WORLD!", 2),
            ("\"HELLO\"", 1),
            ("(HELLO)", 1),
            ("HELLO/WORLD", 1),
            ("A-B-C", 1),
            ("NASA's ROCKET", 2),
            ("DON'T STOP", 3),
            ("CAN'T", 2),
            ("WE'RE HERE", 3),
            ("NASA/ESA", 1),
            ("NASA_ESA", 1),
            ("HELLO,WORLD", 2),
        ];

        for (text, expected) in cases {
            assert_eq!(count_capital_words(text), expected, "input: {text}");
        }
    }

    #[test]
    fn counts_sentences_like_reference_samples() {
        let cases = [
            ("Hello world.", 1),
            ("Dr. Smith went home. He slept.", 2),
            ("U.S.A. is here. Another sent.", 2),
            ("Wait... what? Fine!", 2),
            ("No punctuation", 1),
            ("\"Quoted sentence.\" Next.", 2),
            ("P.S. hello. Another.", 3),
        ];

        for (text, expected) in cases {
            assert_eq!(count_sentences(text), expected, "input: {text}");
        }
    }
}
