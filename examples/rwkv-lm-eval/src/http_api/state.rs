use crate::db::Db;

#[derive(Clone)]
pub struct AppState {
    pub(crate) db: Db,
}

impl AppState {
    pub fn new(db: Db) -> Self {
        Self { db }
    }
}
