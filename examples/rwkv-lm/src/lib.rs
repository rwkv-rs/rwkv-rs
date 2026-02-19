extern crate derive_new;

#[cfg(feature = "training")]
pub mod data;
#[cfg(feature = "inferring")]
pub mod inferring;
pub mod model;
#[cfg(feature = "pth2mpk")]
pub mod pth2mpk;
#[cfg(feature = "training")]
pub mod training;
