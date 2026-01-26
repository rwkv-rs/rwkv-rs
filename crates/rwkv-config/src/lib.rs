pub mod raw;
pub mod validated;

use std::{fs, mem::size_of, path::Path};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub mod config_builder_helpers {
    /// 辅助trait：智能转换 T 或 Option<T> 到 Option<T>
    pub trait IntoBuilderOption<T> {
        fn into_builder_option(self) -> Option<T>;
    }

    impl<T> IntoBuilderOption<T> for T {
        fn into_builder_option(self) -> Option<T> {
            Some(self)
        }
    }

    impl<T> IntoBuilderOption<T> for Option<T> {
        fn into_builder_option(self) -> Option<T> {
            self
        }
    }
}

pub fn load_toml<P: AsRef<Path>, T: DeserializeOwned + 'static>(path: P) -> T {
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read file at path: {}", path.as_ref().display()));

    toml::from_str(&content)
        .unwrap_or_else(|_| panic!("Invalid TOML format in file: {}", path.as_ref().display()))
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelTypeOptions {
    AutoRegressive,
    SequenceEmbedding,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainStageOptions {
    #[default]
    Pretrain,
    Distillation,
    StateTuning,
    SftMiss,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerOptions {
    #[default]
    AdamW,
    Muon,
    Adopt,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFormatOptions {
    #[default]
    Rwkv,
    RwkvLegacy,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum TokenUnitDType {
    U8 = 0,
    #[default]
    U16 = 1,
    F32 = 2,
}

impl TokenUnitDType {
    pub fn is_discrete(&self) -> bool {
        matches!(self, TokenUnitDType::U8 | TokenUnitDType::U16)
    }

    pub fn get_dtype(code: u8) -> Self {
        match code {
            0 => TokenUnitDType::U8,
            1 => TokenUnitDType::U16,
            2 => TokenUnitDType::F32,
            _ => panic!("Unsupported DTYPE code: {}", code),
        }
    }

    pub fn get_token_unit_size(&self) -> usize {
        match self {
            TokenUnitDType::U8 => size_of::<u8>(),
            TokenUnitDType::U16 => size_of::<u16>(),
            TokenUnitDType::F32 => size_of::<f32>(),
        }
    }
}
