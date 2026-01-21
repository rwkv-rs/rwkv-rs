pub mod raw;
pub mod validated;

use std::{fs, path::Path};

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
