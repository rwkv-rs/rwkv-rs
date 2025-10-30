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
