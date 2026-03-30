mod health;
mod index;
mod openapi;

pub(crate) use health::{__path_health, health};
pub(crate) use index::{__path_index, index};
pub(crate) use openapi::{__path_openapi_json, openapi_json};
