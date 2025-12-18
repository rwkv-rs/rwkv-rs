use burn::{
    module::{AutodiffModule, ModuleVisitor, Param, ParamId},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};

/// A simple struct to hold the categorized parameter IDs.
#[derive(Debug, Clone, Default)]
pub struct ParamGroups {
    pub high_lr: Vec<ParamId>,
    pub with_wd: Vec<ParamId>,
    pub no_wd: Vec<ParamId>,
}

/// A visitor that traverses a model and categorizes its parameter IDs based on
/// predefined rules.
pub struct ParamGrouperVisitor<'a> {
    groups: &'a mut ParamGroups,
}

impl ParamGrouperVisitor<'_> {
    /// Creates a new `ParamGroups` container by visiting the given model.
    /// This is the main entry point for the grouping logic.
    pub fn group_params<B, M>(model: &M) -> ParamGroups
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        let mut groups = ParamGroups::default();

        let mut visitor = ParamGrouperVisitor {
            groups: &mut groups,
        };

        model.visit(&mut visitor);

        groups
    }
}

impl<B: Backend> ModuleVisitor<B> for ParamGrouperVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let name = param.id.to_string();

        let rank = D;

        // Grouping logic based on parameter name and rank
        // 1. Special parameters (e.g., LoRA bias) get higher learning rate
        if name.contains("param_weight_decay_lora.bias") {
            self.groups.high_lr.push(param.id);
        }
        // 2. Weight matrices (rank >= 2) get weight decay
        else if name.ends_with(".weight") && rank >= 2 {
            self.groups.with_wd.push(param.id);
        }
        // 3. Everything else (biases, layer norms, etc.) without weight decay
        else {
            self.groups.no_wd.push(param.id);
        }
    }
}
