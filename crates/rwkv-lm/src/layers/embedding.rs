use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    prelude::{Backend, Float, Int, Tensor},
};
use rwkv_data::mmap::dtype::TokenUnit;

#[derive(Config, Debug)]
pub struct EmbModuleConfig {
    vocabulary_size: usize,
    embedded_dim: usize,
}

impl EmbModuleConfig {
    pub fn init<B: Backend, T: TokenUnit>(&self, device: &B::Device) -> EmbModule<B> {
        if T::IS_DISCRETE {
            EmbModule::Discrete(
                EmbeddingConfig::new(self.vocabulary_size, self.embedded_dim).init(device),
            )
        } else {
            EmbModule::Continuous(
                LinearConfig::new(self.vocabulary_size, self.embedded_dim).init(device),
            )
        }
    }
}

#[derive(Module, Debug)]
pub enum EmbModule<B: Backend> {
    Discrete(Embedding<B>),
    Continuous(Linear<B>),
}

impl<B: Backend> EmbModule<B> {
    pub fn forward(&self, x: TokensOptions<B>) -> Tensor<B, 3, Float> {
        match self {
            EmbModule::Discrete(embed) => embed.forward(match x {
                TokensOptions::SingleUnitIntTokens(t) => t,
                _ => panic!("Unsupported tokens received"),
            }),
            EmbModule::Continuous(embed) => embed.forward(match x {
                TokensOptions::SingleUnitFloatTokens(t) => t.unsqueeze_dim(2),
                TokensOptions::MultiUnitFloatTokens(t) => t,
                _ => panic!("Unsupported tokens received"),
            }),
        }
    }

    pub fn weight(&self) -> Tensor<B, 2, Float> {
        match self {
            EmbModule::Discrete(embed) => embed.weight.val(),
            EmbModule::Continuous(embed) => embed.weight.val(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum TokensOptions<B: Backend> {
    SingleUnitIntTokens(Tensor<B, 2, Int>),
    SingleUnitFloatTokens(Tensor<B, 2, Float>),
    MultiUnitIntTokens(Tensor<B, 3, Int>),
    MultiUnitFloatTokens(Tensor<B, 3, Float>),
}

impl<B: Backend> TokensOptions<B> {
    pub fn device(&self) -> B::Device {
        match self {
            TokensOptions::SingleUnitIntTokens(tensor) => tensor.device(),
            TokensOptions::SingleUnitFloatTokens(tensor) => tensor.device(),
            TokensOptions::MultiUnitIntTokens(tensor) => tensor.device(),
            TokensOptions::MultiUnitFloatTokens(tensor) => tensor.device(),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::utils::test_tools::*;

    #[test]
    fn test_embed() {
        let device = &get_test_device::<TestBackend>();
        let model = get_test_model(device);

        let input = load_expected_i64::<TestBackend, 2>("input", device);

        let input_options = TokensOptions::SingleUnitIntTokens(input);

        let embed_output = model.embed.forward(input_options);

        // let embed_weight = model.embed.weight().to_data().to_vec::<f32>().unwrap();
        // let len = embed_weight.len().min(50);
        // println!("embed_weight: {:?}", &embed_weight[0..len]);
        // println!("embed_output: {:?}",
        // embed_output.to_data().to_vec::<f32>().unwrap());
        let expected_embed_output = load_expected_f32::<TestBackend, 3>("emb_output", device);

        assert_closeness(
            &embed_output,
            &expected_embed_output,
            "model.embed",
            MIN_PASS_RATE,
            RELATIVE_ERROR,
        );
    }
}
