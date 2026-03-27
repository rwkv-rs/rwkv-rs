use rwkv_config::raw::eval::SpaceDbConfig;

use crate::db::{Db, connect};

pub(crate) async fn connect_db_if_configured(
    upload_to_space: bool,
    cfg: Option<&SpaceDbConfig>,
    max_connections: u32,
) -> Option<Db> {
    if !upload_to_space {
        println!("database persistence: disabled");
        return None;
    }

    let Some(cfg) = cfg else {
        panic!("upload_to_space=true requires [space_db] config");
    };
    if !is_space_db_configured(cfg) {
        panic!("upload_to_space=true requires a complete [space_db] config");
    }

    let db = connect(cfg, max_connections)
        .await
        .unwrap_or_else(|err| panic!("failed to connect to postgres: {err}"));
    println!("database persistence: enabled (pool max connections = {max_connections})");
    Some(db)
}

fn is_space_db_configured(cfg: &SpaceDbConfig) -> bool {
    !cfg.host.trim().is_empty()
        && !cfg.username.trim().is_empty()
        && !cfg.password.trim().is_empty()
        && !cfg.port.trim().is_empty()
        && !cfg.database_name.trim().is_empty()
}
