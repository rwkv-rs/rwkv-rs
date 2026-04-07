use rwkv_config::validated::eval::FinalEvalConfig;

const DEFAULT_JUDGER_CONCURRENCY: usize = 8;
const DEFAULT_CHECKER_CONCURRENCY: usize = 8;
const DEFAULT_DB_POOL_MAX_CONNECTIONS: u32 = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RunMode {
    #[default]
    New,
    Resume,
    Rerun,
}

impl RunMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "new" => Ok(Self::New),
            "resume" => Ok(Self::Resume),
            "rerun" => Ok(Self::Rerun),
            other => Err(format!(
                "unsupported run mode `{other}`; expected one of: new, resume, rerun"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Resume => "resume",
            Self::Rerun => "rerun",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EvaluatingOptions {
    pub run_mode: RunMode,
    pub skip_checker: bool,
    pub skip_dataset_check: bool,
    pub judger_concurrency: usize,
    pub checker_concurrency: usize,
    pub db_pool_max_connections: u32,
}

impl Default for EvaluatingOptions {
    fn default() -> Self {
        Self {
            run_mode: RunMode::New,
            skip_checker: false,
            skip_dataset_check: false,
            judger_concurrency: DEFAULT_JUDGER_CONCURRENCY,
            checker_concurrency: DEFAULT_CHECKER_CONCURRENCY,
            db_pool_max_connections: DEFAULT_DB_POOL_MAX_CONNECTIONS,
        }
    }
}

pub(crate) fn build_evaluating_options(eval_cfg: &FinalEvalConfig) -> EvaluatingOptions {
    let run_mode = RunMode::parse(&eval_cfg.run_mode)
        .unwrap_or_else(|err| panic!("invalid `run_mode` in eval config: {err}"));
    assert!(
        eval_cfg.judger_concurrency > 0,
        "`judger_concurrency` must be > 0"
    );
    if !eval_cfg.skip_checker {
        assert!(
            eval_cfg.checker_concurrency > 0,
            "`checker_concurrency` must be > 0 when checker is enabled"
        );
    }
    assert!(
        eval_cfg.db_pool_max_connections > 0,
        "`db_pool_max_connections` must be > 0"
    );

    EvaluatingOptions {
        run_mode,
        skip_checker: eval_cfg.skip_checker,
        skip_dataset_check: eval_cfg.skip_dataset_check,
        judger_concurrency: eval_cfg.judger_concurrency,
        checker_concurrency: eval_cfg.checker_concurrency,
        db_pool_max_connections: eval_cfg.db_pool_max_connections,
    }
}
