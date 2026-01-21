//! RWKV 总入口（WIP）。
//!
//! 该 crate 未来将负责统一 feature 开关并重导出下游 API，用户只需依赖这一入口。
//! This crate is the RWKV facade (WIP), unifying features and re-exporting downstream APIs.

pub mod custom {
    pub use burn::*;
}

#[macro_export]
macro_rules! custom_mode {
    () => {
        use $crate::custom as burn;
    };
}

pub mod config {
    pub use rwkv_config::*;
}

pub mod data {
    pub use rwkv_data::*;
}

pub mod lm {
    pub use rwkv_lm::*;
}

pub mod train {
    pub use rwkv_train::*;
}