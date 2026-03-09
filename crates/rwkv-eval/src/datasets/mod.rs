pub mod coding;
pub mod function_calling;
pub mod hf_downloader;
pub mod hf_viewer;
pub mod instruction_following;
pub mod knowledge;
pub mod maths;
pub mod parquet_utils;


pub trait Benchmark: Send + Sync {
    type Item;

    fn check(&self) -> bool;  // return need_download
    fn download(&self);
    fn load(&mut self);
    fn get_expected_context(&self, item: Self::Item) -> String;
    fn get_ref_answer(&self, item: Self::Item) -> String;
    fn get_evaluator(&self);
}
