pub mod coding;
pub mod function_calling;
pub mod hf_downloader;
pub(crate) mod hf_viewer;
pub mod instruction_following;
pub mod knowledge;
pub mod maths;
pub(crate) mod parquet_utils;

use std::collections::HashSet;
use std::fs::remove_dir_all;
use std::path::{Path, PathBuf};

use crate::error::BenchmarkError;
use crate::evaluators::Evaluator;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkSplit {
    Train,
    Dev,
    Validation,
    Test,
}

impl BenchmarkSplit {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Dev => "dev",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }
}

pub trait Benchmark: Send + Sync {
    fn name(&self) -> &'static str;
    fn dataset_root(&self) -> &Path;
    fn dataset_dir(&self) -> PathBuf;
    fn avg_k(&self) -> usize;
    fn pass_k(&self) -> usize;
    fn with_llm_judger(&self) -> bool;
    fn download(&self);
    fn load(&mut self) -> Result<(), BenchmarkError>;
    fn check(&self) -> Result<(), BenchmarkError>;
    fn splits(&self) -> &'static [BenchmarkSplit];
    fn len(&self, split: BenchmarkSplit) -> usize;
    fn get_expected_context(&self, split: BenchmarkSplit, index: usize) -> String;
    fn get_ref_answer(&self, split: BenchmarkSplit, index: usize) -> String;
    fn get_evaluator(&self) -> Box<dyn Evaluator>;
}

pub fn expand_benchmark_names(
    benchmark_fields: &[String],
    extra_benchmark_names: &[String],
) -> Result<Vec<String>, BenchmarkError> {
    let mut names = Vec::new();
    let mut seen = HashSet::new();

    for field in benchmark_fields {
        let expanded = match canonical_name(field).as_str() {
            "knowledge" => vec!["mmlu"],
            "math" | "maths" => Vec::new(),
            "coding" => Vec::new(),
            "instructionfollowing" => Vec::new(),
            "functioncall" | "functioncalling" => Vec::new(),
            _ => return Err(BenchmarkError::UnsupportedField(field.clone())),
        };

        for name in expanded {
            if seen.insert(name.to_string()) {
                names.push(name.to_string());
            }
        }
    }

    for name in extra_benchmark_names {
        let canonical = canonical_name(name);
        if seen.insert(canonical.clone()) {
            names.push(canonical);
        }
    }

    Ok(names)
}

pub fn build_benchmark<P: AsRef<Path>>(
    benchmark_name: &str,
    dataset_root: P,
) -> Result<Box<dyn Benchmark>, BenchmarkError> {
    match canonical_name(benchmark_name).as_str() {
        "mmlu" => Ok(Box::new(knowledge::mmlu::Mmlu::new(dataset_root.as_ref()))),
        other => Err(BenchmarkError::UnsupportedBenchmark(other.to_string())),
    }
}

pub fn ensure_dataset_ready(benchmark: &mut dyn Benchmark) -> Result<(), BenchmarkError> {
    if benchmark.load().is_ok() && benchmark.check().is_ok() {
        return Ok(());
    }

    let dataset_dir = benchmark.dataset_dir();
    if dataset_dir.exists() {
        remove_dir_all(&dataset_dir)?;
    }

    benchmark.download();
    benchmark.load()?;
    benchmark.check()?;
    Ok(())
}

fn canonical_name(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::expand_benchmark_names;

    #[test]
    fn expand_benchmark_names_keeps_registered_order() {
        let names = expand_benchmark_names(
            &["Knowledge".to_string(), "Function Call".to_string()],
            &["mmlu".to_string()],
        )
        .unwrap();

        assert_eq!(names, vec!["mmlu"]);
    }
}
