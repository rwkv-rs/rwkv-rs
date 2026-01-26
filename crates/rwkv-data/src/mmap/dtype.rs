use burn::tensor::Element;
use bytemuck::Pod;
pub use rwkv_config::TokenUnitDType;
use serde::Serialize;

pub trait TokenUnit: IsDiscrete + Clone + Serialize + Pod + Element {
    const DTYPE: TokenUnitDType;
}

impl TokenUnit for u8 {
    const DTYPE: TokenUnitDType = TokenUnitDType::U8;
}

impl TokenUnit for u16 {
    const DTYPE: TokenUnitDType = TokenUnitDType::U16;
}

impl TokenUnit for f32 {
    const DTYPE: TokenUnitDType = TokenUnitDType::F32;
}

pub trait IsDiscrete {
    const IS_DISCRETE: bool;
}

impl IsDiscrete for u8 {
    const IS_DISCRETE: bool = true;
}

impl IsDiscrete for u16 {
    const IS_DISCRETE: bool = true;
}

impl IsDiscrete for f32 {
    const IS_DISCRETE: bool = false;
}
