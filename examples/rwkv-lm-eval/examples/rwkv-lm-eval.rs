use std::path::PathBuf;

use rwkv_config::get_arg_value;
use rwkv_eval::init::init_cfg;
use rwkv_lm_eval::evaluating::evaluating;
use rwkv_lm_eval::paths;


#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_dir = get_arg_value(&args, "--config-dir")
        .map(PathBuf::from)
        .unwrap_or_else(paths::config_dir);
    let eval_cfg_name = get_arg_value(&args, "--eval-config").unwrap_or_else(|| "example".into());

    let eval_cfg_builder = init_cfg(&config_dir, &eval_cfg_name);

    println!(
        "eval cfg: {eval_cfg_name} (config_dir: {})",
        config_dir = config_dir.display(),
    );
    println!("datasets dir: {}", paths::datasets_path().display());

    let report = evaluating(&eval_cfg_builder, paths::datasets_path())
        .await
        .unwrap_or_else(|error| panic!("failed to run eval: {error}"));

    println!("experiment: {}", report.experiment_name);
    for model_run in report.model_runs {
        println!("model: {}", model_run.model_name);
        for summary in model_run.benchmark_summaries {
            println!(
                "  {} -> passed {}/{} (failed {}, accuracy {:.4}, avg_k {}, pass_k {})",
                summary.benchmark_name,
                summary.passed,
                summary.total,
                summary.failed,
                summary.accuracy,
                summary.avg_k,
                summary.pass_k,
            );
        }
    }
}
