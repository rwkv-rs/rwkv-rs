#[cfg(feature = "wgpu")]
pub(crate) use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
#[cfg(not(feature = "wgpu"))]
pub(crate) use burn_cubecl::{BoolElement, CubeBackend, FloatElement, IntElement};

pub(crate) use burn_cubecl::{CubeElement, CubeRuntime};
