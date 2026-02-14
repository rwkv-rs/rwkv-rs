use burn::cubecl;
use cubecl::{cube, prelude::*};

const W_SCALE: f32 = -0.6065306597; // -exp(-0.5), matches RWKV7 clampw CUDA kernel

#[cube(launch)]
pub fn wkv7_infer_forward_kernel<F: Float>(
    inputs: &Wkv7InferForwardInputs<F>,
    outputs: &mut Wkv7InferForwardOutputs<F>,
    #[comptime] config: Wkv7InferConfig,
) {
    let context_length = comptime![config.context_length];
    let num_heads = comptime![config.num_heads];
    let head_size = comptime![config.head_size];

    let batch_index = CUBE_POS_Y as usize;
    let head_index = CUBE_POS_X as usize;
    let head_dim_index = UNIT_POS as usize;

    if head_dim_index >= head_size {
        terminate!();
    }

    let mut state = Array::<F>::new(head_size);
    let initial_state_base = (batch_index * num_heads + head_index) * head_size * head_size
        + head_dim_index * head_size;

    #[unroll(true)]
    for i in 0..head_size {
        state[i] = inputs.initial_state[initial_state_base + i];
    }

    let mut shared_receptance = SharedMemory::<F>::new(head_size);
    let mut shared_weight_decay = SharedMemory::<F>::new(head_size);
    let mut shared_key = SharedMemory::<F>::new(head_size);
    let mut shared_value = SharedMemory::<F>::new(head_size);
    let mut shared_removal = SharedMemory::<F>::new(head_size);
    let mut shared_replacement = SharedMemory::<F>::new(head_size);

    let batch_head_base = (batch_index * context_length * num_heads + head_index) * head_size;
    let context_mask_base = batch_index * context_length;

    for t in 0..context_length {
        let flat_index = batch_head_base + t * num_heads * head_size + head_dim_index;
        let context_mask = inputs.context_mask[context_mask_base + t];

        if context_mask == F::new(0.0) {
            outputs.output[flat_index] = F::new(0.0);
        } else {
            sync_cube();

            shared_receptance[head_dim_index] = inputs.receptance[flat_index];
            shared_key[head_dim_index] = inputs.key[flat_index];
            shared_removal[head_dim_index] = inputs.removal[flat_index];
            shared_replacement[head_dim_index] = inputs.replacement[flat_index];
            shared_value[head_dim_index] = inputs.value[flat_index];
            let w_raw = inputs.weight_decay[flat_index];
            let w_sig = F::new(1.0) / (F::new(1.0) + F::exp(-w_raw));
            shared_weight_decay[head_dim_index] = F::exp(F::new(W_SCALE) * w_sig);
            sync_cube();

            let mut removal_state = F::new(0.0);
            #[unroll(true)]
            for i in 0..head_size {
                removal_state += shared_removal[i] * state[i];
            }

            let v = shared_value[head_dim_index];
            let mut y = F::new(0.0);

            #[unroll(true)]
            for i in 0..head_size {
                state[i] = state[i] * shared_weight_decay[i]
                    + removal_state * shared_replacement[i]
                    + shared_key[i] * v;
                y += state[i] * shared_receptance[i];
            }

            outputs.output[flat_index] = y;
        }
    }

    let final_state_base = (batch_index * num_heads + head_index) * head_size * head_size
        + head_dim_index * head_size;

    #[unroll(true)]
    for i in 0..head_size {
        outputs.final_state[final_state_base + i] = state[i];
    }
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7InferForwardInputs<F: Float> {
    pub weight_decay: Tensor<F>,
    pub receptance: Tensor<F>,
    pub key: Tensor<F>,
    pub value: Tensor<F>,
    pub removal: Tensor<F>,
    pub replacement: Tensor<F>,
    pub initial_state: Tensor<F>,
    pub context_mask: Tensor<F>,
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7InferForwardOutputs<F: Float> {
    pub output: Tensor<F>,
    pub final_state: Tensor<F>,
}

#[derive(CubeLaunch, CubeType, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Wkv7InferConfig {
    pub context_length: usize,
    pub num_heads: usize,
    pub head_size: usize,
}
