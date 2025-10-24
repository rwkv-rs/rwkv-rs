use burn::{
    config::Config, lr_scheduler::LrScheduler, optim::LearningRate, tensor::backend::Backend,
};

#[derive(Config, Debug)]

pub struct WsdLrSchedulerConfig {
    lr_init: LearningRate,
    lr_final: LearningRate,
    warmup_steps: usize,
    num_steps: usize,
}

impl WsdLrSchedulerConfig {
    pub fn init(&self) -> WsdLrScheduler {
        assert!(
            self.lr_init > 0. && self.lr_init <= 1.,
            "Initial learning rate must be greater than 0 and at most 1, got {}",
            self.lr_init
        );

        assert!(
            self.lr_final >= 0.0 && self.lr_final <= self.lr_init,
            "Final learning rate must be at least 0 and at most equal to the initial learning rate, got lr_final={}, lr_init={}",
            self.lr_final,
            self.lr_init
        );

        assert!(
            self.num_steps > 0,
            "Total steps must be at least 1, got {}",
            self.num_steps
        );

        assert!(
            self.warmup_steps <= self.num_steps,
            "Warmup steps must not exceed total steps, got warmup_steps={}, total_steps={}",
            self.warmup_steps,
            self.num_steps
        );

        WsdLrScheduler {
            lr_init: self.lr_init,
            lr_final: self.lr_final,
            warmup_steps: self.warmup_steps,
            num_steps: self.num_steps,
            step_index: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WsdLrScheduler {
    lr_init: LearningRate,
    lr_final: LearningRate,
    warmup_steps: usize,
    num_steps: usize,
    step_index: usize,
}

impl LrScheduler for WsdLrScheduler {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        self.step_index += 1;

        let step_index = self.step_index as f64;

        let warmup_steps = self.warmup_steps as f64;

        // Warmup phase: linear increase
        if step_index < warmup_steps {
            return self.lr_init * step_index / warmup_steps;
        }

        // Final phase: maintain lr_final after total_steps
        if self.step_index > self.num_steps {
            return self.lr_final;
        }

        // Cosine decay phase
        let decay_total_steps = (self.num_steps - self.warmup_steps) as f64;

        if decay_total_steps <= 0.0 {
            return self.lr_final;
        }

        let decay_current_step = (step_index - warmup_steps).min(decay_total_steps);

        let progress = decay_current_step / decay_total_steps;

        let cos_decay = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());

        self.lr_final + (self.lr_init - self.lr_final) * cos_decay
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.step_index
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.step_index = record;

        self
    }
}
