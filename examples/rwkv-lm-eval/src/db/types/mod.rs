mod commands;
mod queries;
mod records;
mod status;

use sqlx::PgPool;

pub use commands::*;
pub use queries::*;
pub use records::*;
pub use status::*;

#[derive(Clone)]
pub struct Db {
    pub(crate) pool: PgPool,
}
