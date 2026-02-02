use burn::cubecl;
use cubecl::{cube, prelude::*};

const W_SCALE: f32 = -0.6065306597; // -exp(-0.5), matches RWKV7 clampw CUDA kernel

#[cube(launch)]
pub fn wkv7_forward_kernel<F: Float>(
    inputs: &Wkv7ForwardInputs<F>,
    outputs: &mut Wkv7ForwardOutputs<F>,
    #[comptime] config: Wkv7Config,
) {
    let sequence_length = comptime![config.sequence_length];
    let num_heads = comptime![config.num_heads];
    let head_size = comptime![config.head_size];
    let chunk_length = comptime![config.chunk_length];
    let use_initial_state = comptime![config.use_initial_state] != 0;
    let return_final_state = comptime![config.return_final_state] != 0;

    let batch_index = CUBE_POS_Y as usize;
    let head_index = CUBE_POS_X as usize;
    let head_dim_index = UNIT_POS as usize;

    if head_dim_index >= head_size {
        terminate!();
    }

    let mut state = Array::<F>::new(head_size);

    if use_initial_state {
        let initial_state_base = (batch_index * num_heads + head_index) * head_size * head_size
            + head_dim_index * head_size;

        #[unroll(true)]
        for i in 0..head_size {
            state[i] = inputs.initial_state[initial_state_base + i];
        }
    } else {
        #[unroll(true)]
        for i in 0..head_size {
            state[i] = F::new(0.0);
        }
    }

    let mut shared_receptance = SharedMemory::<F>::new(head_size);
    let mut shared_weight_decay = SharedMemory::<F>::new(head_size);
    let mut shared_key = SharedMemory::<F>::new(head_size);
    let mut shared_value = SharedMemory::<F>::new(head_size);
    let mut shared_removal = SharedMemory::<F>::new(head_size);
    let mut shared_replacement = SharedMemory::<F>::new(head_size);

    for t in 0..sequence_length {
        let flat_index = batch_index * sequence_length * num_heads * head_size
            + t * num_heads * head_size
            + head_index * head_size
            + head_dim_index;
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
        outputs.removal_state[flat_index] = removal_state;

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

        if (t + 1) % chunk_length == 0 {
            let base = (batch_index * num_heads + head_index)
                * (sequence_length / chunk_length)
                * head_size
                * head_size
                + (t / chunk_length) * head_size * head_size
                + head_dim_index;

            #[unroll(true)]
            for i in 0..head_size {
                outputs.state[base + i * head_size] = state[i];
            }
        }
    }

    if return_final_state {
        let final_state_base = (batch_index * num_heads + head_index) * head_size * head_size
            + head_dim_index * head_size;

        #[unroll(true)]
        for i in 0..head_size {
            outputs.final_state[final_state_base + i] = state[i];
        }
    }
}

#[cube(launch)]
pub fn wkv7_backward_kernel<F: Float>(
    inputs: &Wkv7BackwardInputs<F>,
    outputs: &mut Wkv7BackwardOutputs<F>,
    #[comptime] config: Wkv7Config,
) {
    let sequence_length = comptime![config.sequence_length];
    let num_heads = comptime![config.num_heads];
    let head_size = comptime![config.head_size];
    let chunk_length = comptime![config.chunk_length];
    let state_line_size = comptime![config.state_line_size];
    let use_final_state_grad = comptime![config.use_final_state_grad] != 0;
    let write_initial_state_grad = comptime![config.write_initial_state_grad] != 0;

    let batch_index = CUBE_POS_Y as usize;
    let head_index = CUBE_POS_X as usize;
    let head_dim_index = UNIT_POS as usize;

    if head_dim_index >= head_size {
        terminate!();
    }

    let mut state_transposed = Array::<F>::new(head_size);
    let mut state_grad = Array::<F>::new(head_size);
    let mut state_transposed_grad = Array::<F>::new(head_size);

    #[unroll(true)]
    for i in 0..head_size {
        state_transposed[i] = F::new(0.0);
    }

    if use_final_state_grad {
        let final_state_base =
            (batch_index * num_heads + head_index) * head_size * head_size;
        let row_base = final_state_base + head_dim_index * head_size;
        let col_base = final_state_base + head_dim_index;

        #[unroll(true)]
        for i in 0..head_size {
            state_grad[i] = inputs.final_state_grad[row_base + i];
            state_transposed_grad[i] = inputs.final_state_grad[col_base + i * head_size];
        }
    } else {
        #[unroll(true)]
        for i in 0..head_size {
            state_grad[i] = F::new(0.0);
            state_transposed_grad[i] = F::new(0.0);
        }
    }

    let mut shared_receptance = SharedMemory::<F>::new(head_size);
    let mut shared_weight_decay = SharedMemory::<F>::new(head_size);
    let mut shared_key = SharedMemory::<F>::new(head_size);
    let mut shared_value = SharedMemory::<F>::new(head_size);
    let mut shared_removal = SharedMemory::<F>::new(head_size);
    let mut shared_replacement = SharedMemory::<F>::new(head_size);
    let mut shared_output_grad = SharedMemory::<F>::new(head_size);
    let mut shared_removed_state = SharedMemory::<F>::new(head_size);
    let mut shared_replaced_state_grad = SharedMemory::<F>::new(head_size);

    let t = RuntimeCell::<usize>::new(sequence_length);

    while t.read() > 0 {
        t.store(t.read() - 1);

        let flat_index = batch_index * sequence_length * num_heads * head_size
            + t.read() * num_heads * head_size
            + head_index * head_size
            + head_dim_index;
        sync_cube();

        let receptance_indexed = inputs.receptance[flat_index];
        shared_receptance[head_dim_index] = receptance_indexed;

        let raw_w = inputs.weight_decay[flat_index];
        let w_sig = F::new(1.0) / (F::new(1.0) + F::exp(-raw_w));
        let weight_decay_indexed = F::exp(F::new(W_SCALE) * w_sig);
        shared_weight_decay[head_dim_index] = weight_decay_indexed;

        let key_indexed = inputs.key[flat_index];
        shared_key[head_dim_index] = key_indexed;
        let removal_indexed = inputs.removal[flat_index];

        shared_removal[head_dim_index] = removal_indexed;
        let replacement_indexed = inputs.replacement[flat_index];

        shared_replacement[head_dim_index] = replacement_indexed;
        shared_value[head_dim_index] = inputs.value[flat_index];

        let output_grad_indexed = inputs.output_grad[flat_index];
        shared_output_grad[head_dim_index] = output_grad_indexed;

        shared_removed_state[head_dim_index] = inputs.removal_state[flat_index];
        sync_cube();

        if (t.read() + 1) % chunk_length == 0 {
            let base = (batch_index * num_heads + head_index)
                * (sequence_length / chunk_length)
                * head_size
                * head_size
                + (t.read() / chunk_length) * head_size * head_size
                + head_dim_index * head_size;
            let line_base = base / state_line_size;

            #[unroll(true)]
            for j in 0..(head_size / state_line_size) {
                let line = inputs.state[line_base + j];
                #[unroll(true)]
                for k in 0..state_line_size {
                    let idx = j * state_line_size + k;
                    state_transposed[idx] = line[k];
                }
            }
        }

        let mut receptance_grad = F::new(0.0);

        #[unroll(true)]
        for i in 0..head_size {
            receptance_grad += state_transposed[i] * shared_output_grad[i];
        }

        outputs.receptance_grad[flat_index] = receptance_grad;

        let inverse_weight_decay_indexed = F::new(1.0) / weight_decay_indexed;

        #[unroll(true)]
        for i in 0..head_size {
            state_transposed[i] = (state_transposed[i]
                - key_indexed * shared_value[i]
                - replacement_indexed * shared_removed_state[i])
                * inverse_weight_decay_indexed;

            state_grad[i] += output_grad_indexed * shared_receptance[i];
            state_transposed_grad[i] += receptance_indexed * shared_output_grad[i];
        }

        let mut weight_decay_grad = F::new(0.0);
        let mut key_grad = F::new(0.0);
        let mut value_grad = F::new(0.0);
        let mut replacement_grad = F::new(0.0);
        let mut replaced_state_grad = F::new(0.0);

        #[unroll(true)]
        for i in 0..head_size {
            weight_decay_grad += state_transposed_grad[i] * state_transposed[i];
            key_grad += state_transposed_grad[i] * shared_value[i];
            value_grad += state_grad[i] * shared_key[i];
            replaced_state_grad += state_grad[i] * shared_replacement[i];
            replacement_grad += state_transposed_grad[i] * shared_removed_state[i];
        }

        outputs.weight_decay_grad[flat_index] = F::new(W_SCALE)
            * weight_decay_grad
            * weight_decay_indexed
            * w_sig
            * (F::new(1.0) - w_sig);
        outputs.key_grad[flat_index] = key_grad;
        outputs.value_grad[flat_index] = value_grad;
        outputs.replacement_grad[flat_index] = replacement_grad;
        sync_cube();

        shared_replaced_state_grad[head_dim_index] = replaced_state_grad;
        sync_cube();

        let mut da = F::new(0.0);

        #[unroll(true)]
        for i in 0..head_size {
            da += state_transposed[i] * shared_replaced_state_grad[i];
        }

        outputs.removal_grad[flat_index] = da;

        #[unroll(true)]
        for i in 0..head_size {
            state_grad[i] =
                state_grad[i] * shared_weight_decay[i] + replaced_state_grad * shared_removal[i];

            state_transposed_grad[i] = state_transposed_grad[i] * weight_decay_indexed
                + removal_indexed * shared_replaced_state_grad[i];
        }
    }

    if write_initial_state_grad {
        let initial_state_base = (batch_index * num_heads + head_index) * head_size * head_size
            + head_dim_index * head_size;

        #[unroll(true)]
        for i in 0..head_size {
            outputs.initial_state_grad[initial_state_base + i] = state_grad[i];
        }
    }
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7ForwardInputs<F: Float> {
    pub weight_decay: Tensor<F>,
    pub receptance: Tensor<F>,
    pub key: Tensor<F>,
    pub value: Tensor<F>,
    pub removal: Tensor<F>,
    pub replacement: Tensor<F>,
    pub initial_state: Tensor<F>,
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7ForwardOutputs<F: Float> {
    pub state: Tensor<F>,
    pub removal_state: Tensor<F>,
    pub output: Tensor<F>,
    pub final_state: Tensor<F>,
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7BackwardInputs<F: Float> {
    pub weight_decay: Tensor<F>,
    pub receptance: Tensor<F>,
    pub key: Tensor<F>,
    pub value: Tensor<F>,
    pub removal: Tensor<F>,
    pub replacement: Tensor<F>,
    pub state: Tensor<Line<F>>,
    pub removal_state: Tensor<F>,
    pub output_grad: Tensor<F>,
    pub final_state_grad: Tensor<F>,
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7BackwardOutputs<F: Float> {
    pub weight_decay_grad: Tensor<F>,
    pub receptance_grad: Tensor<F>,
    pub key_grad: Tensor<F>,
    pub value_grad: Tensor<F>,
    pub removal_grad: Tensor<F>,
    pub replacement_grad: Tensor<F>,
    pub initial_state_grad: Tensor<F>,
}

#[derive(CubeLaunch, CubeType, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Wkv7Config {
    pub sequence_length: usize,
    pub num_heads: usize,
    pub head_size: usize,
    pub chunk_length: usize,
    pub state_line_size: LineSize,
    // CubeCL does not allow bool as a launchable scalar; use u32 flags for comptime.
    pub use_initial_state: u32,
    pub return_final_state: u32,
    pub use_final_state_grad: u32,
    pub write_initial_state_grad: u32,
}
