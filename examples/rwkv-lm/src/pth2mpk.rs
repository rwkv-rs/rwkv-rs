use crate::model::AutoRegressiveModelConfig;
use rwkv::custom::cubecl::device::DeviceId;
use rwkv::custom::module::Module;
use rwkv::custom::prelude::{Backend, DeviceOps};
use rwkv::custom::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rwkv::custom::store::{ModuleSnapshot, PytorchStore};

#[derive(Clone, Debug)]
pub struct ConvertPthToMpkOptions {
    model_path: String,
    output_path: String,
}

impl ConvertPthToMpkOptions {
    pub fn new(model_path: impl Into<String>, output_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            output_path: output_path.into(),
        }
    }
}

pub fn convert_pth_to_mpk<B: Backend>(
    options: &ConvertPthToMpkOptions,
    model_config: AutoRegressiveModelConfig,
) {
    pth2mpk::<B>(&options.model_path, &options.output_path, model_config);
}

pub fn pth2mpk<B: Backend>(
    model_path: &str,
    output_path: &str,
    model_config: AutoRegressiveModelConfig,
) {
    let mut store = PytorchStore::from_file(model_path)
        .with_key_remapping("^emb\\.weight$", "embed.Discrete.weight")
        .with_key_remapping(
            "^blocks\\.0\\.ln0\\.weight$",
            "layer_norm_for_first_cell.gamma",
        )
        .with_key_remapping(
            "^blocks\\.0\\.ln0\\.bias$",
            "layer_norm_for_first_cell.beta",
        )
        .with_key_remapping("^ln_out\\.weight$", "layer_norm_for_unembed.gamma")
        .with_key_remapping("^ln_out\\.bias$", "layer_norm_for_unembed.beta")
        .with_key_remapping("^head\\.weight$", "unembed.weight")
        .with_key_remapping(
            "blocks.([0-9]+).ln1.weight",
            "cells.$1.pre_layer_norm_for_time_mix.gamma",
        )
        .with_key_remapping(
            "blocks.([0-9]+).ln1.bias",
            "cells.$1.pre_layer_norm_for_time_mix.beta",
        )
        .with_key_remapping(
            "blocks.([0-9]+).ln2.weight",
            "cells.$1.pre_layer_norm_for_channel_mix.gamma",
        )
        .with_key_remapping(
            "blocks.([0-9]+).ln2.bias",
            "cells.$1.pre_layer_norm_for_channel_mix.beta",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.receptance.weight",
            "cells.$1.time_mixer.projection_receptance.weight",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.key.weight",
            "cells.$1.time_mixer.projection_key.weight",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.value.weight",
            "cells.$1.time_mixer.projection_value.weight",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.output.weight",
            "cells.$1.time_mixer.projection_output.weight",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.ln_x.weight",
            "cells.$1.time_mixer.group_norm.gamma",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.ln_x.bias",
            "cells.$1.time_mixer.group_norm.beta",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.k_k",
            "cells.$1.time_mixer.param_key_removal",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.r_k",
            "cells.$1.time_mixer.param_receptance_key_bonus",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.k_a",
            "cells.$1.time_mixer.param_key_replacement",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.w0",
            "cells.$1.time_mixer.param_weight_decay_lora.bias",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.w1",
            "cells.$1.time_mixer.param_weight_decay_lora.w_a",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.w2",
            "cells.$1.time_mixer.param_weight_decay_lora.w_b",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.a0",
            "cells.$1.time_mixer.param_learning_rate_lora.bias",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.a1",
            "cells.$1.time_mixer.param_learning_rate_lora.w_a",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.a2",
            "cells.$1.time_mixer.param_learning_rate_lora.w_b",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.g1",
            "cells.$1.time_mixer.param_output_gate_lora.w_a",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.g2",
            "cells.$1.time_mixer.param_output_gate_lora.w_b",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.v0",
            "cells.$1.time_mixer.param_value_residual_lora.bias",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.v1",
            "cells.$1.time_mixer.param_value_residual_lora.w_a",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.v2",
            "cells.$1.time_mixer.param_value_residual_lora.w_b",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.x_r",
            "cells.$1.time_mixer.param_receptance",
        )
        .with_key_remapping(
            "blocks.([0-9]+).att.x_w",
            "cells.$1.time_mixer.param_weight_decay",
        )
        .with_key_remapping("blocks.([0-9]+).att.x_k", "cells.$1.time_mixer.param_key")
        .with_key_remapping("blocks.([0-9]+).att.x_v", "cells.$1.time_mixer.param_value")
        .with_key_remapping(
            "blocks.([0-9]+).att.x_a",
            "cells.$1.time_mixer.param_learning_rate",
        )
        .with_key_remapping("blocks.([0-9]+).att.x_g", "cells.$1.time_mixer.param_gate")
        .with_key_remapping(
            "blocks.([0-9]+).ffn.key.weight",
            "cells.$1.channel_mixer.w_in.weight",
        )
        .with_key_remapping(
            "blocks.([0-9]+).ffn.value.weight",
            "cells.$1.channel_mixer.w_out.weight",
        )
        .with_key_remapping(
            "blocks.([0-9]+).ffn.x_k",
            "cells.$1.channel_mixer.token_shift_diff_scale",
        )
        .allow_partial(true);

    let device = B::Device::from_id(DeviceId::new(0, 0));

    let mut model = model_config.init::<B>(&device);

    // model.init_weights(&device);
    let _result = model.load_from(&mut store).unwrap();

    // let embed_weight: Vec<f32> = model
    //     .embed
    //     .weight()
    //     .to_data()
    //     .to_vec::<bf16>()
    //     .unwrap()
    //     .iter()
    //     .map(|x| x.to_f32())
    //     .collect();
    //
    // let unembed_weight: Vec<f32> = model
    //     .unembed
    //     .weight
    //     .to_data()
    //     .to_vec::<bf16>()
    //     .unwrap()
    //     .iter()
    //     .map(|x| x.to_f32())
    //     .collect();
    //
    // println!(
    //     "embed_weight min: {}, max: {}, mean: {}",
    //     embed_weight.iter().cloned().fold(f32::INFINITY, f32::min),
    //     embed_weight
    //         .iter()
    //         .cloned()
    //         .fold(f32::NEG_INFINITY, f32::max),
    //     embed_weight.iter().sum::<f32>() / embed_weight.len() as f32
    // );
    //
    // println!(
    //     "unembed_weight min: {}, max: {}, mean: {}",
    //     unembed_weight.iter().cloned().fold(f32::INFINITY, f32::min),
    //     unembed_weight
    //         .iter()
    //         .cloned()
    //         .fold(f32::NEG_INFINITY, f32::max),
    //     unembed_weight.iter().sum::<f32>() / unembed_weight.len() as f32
    // );
    model
        .save_file(
            output_path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Trained model should be saved successfully");
}
