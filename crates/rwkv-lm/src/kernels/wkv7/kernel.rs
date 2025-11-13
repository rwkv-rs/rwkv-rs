use burn::cubecl;
use cubecl::{cube, prelude::*};

#[cube(launch)]
pub fn wkv7_forward_kernel<F: Float>(
    inputs: &Wkv7Inputs<F>,
    outputs: &mut Wkv7Outputs<F>,
    #[comptime] config: Wkv7Config,
) {
    let sequence_length = comptime![config.sequence_length];
    let num_heads = comptime![config.num_heads];
    let head_size = comptime![config.head_size];
    let chunk_length = comptime![config.chunk_length];

    let batch_index = CUBE_POS_Y;
    let head_index = CUBE_POS_X;
    let head_dim_index = UNIT_POS;

    if head_dim_index >= head_size {
        terminate!();
    }

    let mut state = Array::<F>::new(head_size);

    let initial_state_base =
        (batch_index * num_heads + head_index) * head_size * head_size + head_dim_index * head_size;

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
        shared_weight_decay[head_dim_index] = F::exp(-F::exp(inputs.weight_decay[flat_index]));
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

    let batch_index = CUBE_POS_Y;
    let head_index = CUBE_POS_X;
    let head_dim_index = UNIT_POS;

    if head_dim_index >= head_size {
        terminate!();
    }

    let mut state_transposed = Array::<F>::new(head_size);
    let mut state_grad = Array::<F>::new(head_size);
    let mut state_transposed_grad = Array::<F>::new(head_size);

    #[unroll(true)]
    for i in 0..head_size {
        state_transposed[i] = F::new(0.0);
        state_grad[i] = F::new(0.0);
        state_transposed_grad[i] = F::new(0.0);
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

    // let mut receptance_indexed: F = F::new(0.0);
    // let mut weight_decay_indexed: F = F::new(0.0);
    // let mut key_indexed: F = F::new(0.0);
    // let mut removal_indexed: F = F::new(0.0);
    // let mut replacement_indexed: F = F::new(0.0);
    // let mut output_grad_indexed: F = F::new(0.0);
    let t = RuntimeCell::<u32>::new(sequence_length);

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
        let wi_fac = F::exp(raw_w);
        let weight_decay_indexed = F::exp(-wi_fac);
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

            #[unroll(true)]
            for i in 0..head_size {
                state_transposed[i] = inputs.state[base + i];
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

        outputs.weight_decay_grad[flat_index] = -weight_decay_grad * weight_decay_indexed * wi_fac;
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

    let initial_state_base =
        (batch_index * num_heads + head_index) * head_size * head_size + head_dim_index * head_size;

    #[unroll(true)]
    for i in 0..head_size {
        outputs.initial_state_grad[initial_state_base + i] = state_grad[i];
    }
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7Inputs<F: Float> {
    pub weight_decay: Tensor<F>,
    pub receptance: Tensor<F>,
    pub key: Tensor<F>,
    pub value: Tensor<F>,
    pub removal: Tensor<F>,
    pub replacement: Tensor<F>,
    pub initial_state: Tensor<F>,
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7Outputs<F: Float> {
    pub state: Tensor<F>,
    pub removal_state: Tensor<F>,
    pub output: Tensor<F>,
}

#[derive(CubeLaunch, CubeType)]
pub struct Wkv7BackwardInputs<F: Float> {
    pub weight_decay: Tensor<F>,
    pub receptance: Tensor<F>,
    pub key: Tensor<F>,
    pub value: Tensor<F>,
    pub removal: Tensor<F>,
    pub replacement: Tensor<F>,
    pub state: Tensor<F>,
    pub removal_state: Tensor<F>,
    pub output_grad: Tensor<F>,
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
    pub _batch_size: u32,
    pub sequence_length: u32,
    pub num_heads: u32,
    pub head_size: u32,
    pub chunk_length: u32,
}
