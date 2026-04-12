#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

extern crate derive_new;

#[cfg(feature = "training")]
pub mod data;
#[cfg(feature = "inferring")]
pub mod inferring;
pub mod model;
pub mod paths;
#[cfg(feature = "pth2bpk")]
pub mod pth2bpk;
#[cfg(feature = "training")]
pub mod training;
