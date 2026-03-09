pub mod coding;
pub mod function_calling;
pub mod hf_downloader;
pub(crate) mod hf_viewer;
pub mod instruction_following;
pub mod knowledge;
pub mod maths;
pub(crate) mod parquet_utils;

use std::path::{Path, PathBuf};

use crate::error::BenchmarkError;
use crate::evaluators::Evaluator;


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
    fn len(&self, split: BenchmarkSplit) -> usize;
    fn get_expected_context(&self, index: usize) -> String;
    fn get_ref_answer(&self, index: usize) -> String;
    fn get_evaluator(&self) -> Box<dyn Evaluator>;
}
