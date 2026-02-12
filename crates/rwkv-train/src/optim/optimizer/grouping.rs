use burn::{
    module::{AutodiffModule, ModuleVisitor, Param, ParamId},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ParamGroupingMode {
    Default,
    StateTune,
}

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
    grouping_mode: ParamGroupingMode,
    module_path: Vec<String>,
    state_param_count: usize,
}

impl ParamGrouperVisitor<'_> {
    /// Creates a new `ParamGroups` container by visiting the given model.
    /// This is the main entry point for the grouping logic.
    pub fn group_params<B, M>(model: &M, grouping_mode: ParamGroupingMode) -> ParamGroups
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        let mut groups = ParamGroups::default();
        let mut visitor = ParamGrouperVisitor {
            groups: &mut groups,
            grouping_mode,
            module_path: Vec::new(),
            state_param_count: 0,
        };

        model.visit(&mut visitor);

        if grouping_mode == ParamGroupingMode::StateTune && visitor.state_param_count == 0 {
            panic!(
                "StateTune mode did not find any trainable state parameter. \
                 Expected parameter path suffix: state.state."
            );
        }

        groups
    }

    fn is_param_weight_decay_lora_bias(&self) -> bool {
        match self.module_path.as_slice() {
            [.., second_last, last] => second_last == "param_weight_decay_lora" && last == "bias",
            _ => false,
        }
    }

    fn is_param_state_state(&self) -> bool {
        match self.module_path.as_slice() {
            [.., second_last, last] => second_last == "state" && last == "state",
            _ => false,
        }
    }

    fn group_param_default<const D: usize>(&mut self, param_id: ParamId) {
        if self.is_param_weight_decay_lora_bias() {
            self.groups.high_lr.push(param_id);
        } else if self.module_path.last().map(String::as_str) == Some("weight") && D >= 2 {
            self.groups.with_wd.push(param_id);
        } else {
            self.groups.no_wd.push(param_id);
        }
    }

    fn group_param_state_tune<const D: usize>(&mut self, param_id: ParamId) {
        if !self.is_param_state_state() {
            return;
        }

        self.state_param_count += 1;

        if D >= 2 {
            self.groups.with_wd.push(param_id);
        } else {
            self.groups.no_wd.push(param_id);
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for ParamGrouperVisitor<'_> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.module_path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.module_path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        match self.grouping_mode {
            ParamGroupingMode::Default => self.group_param_default::<D>(param.id),
            ParamGroupingMode::StateTune => self.group_param_state_tune::<D>(param.id),
        }
    }
}
