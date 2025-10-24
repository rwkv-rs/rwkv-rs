use std::path::Path;

use redb::{Database, ReadableDatabase, TableDefinition};
use xxhash_rust::xxh3::xxh3_128;

const MAP_TABLE: TableDefinition<u128, (u64, u64)> = TableDefinition::new("map");

pub struct Map {
    db: Database,
}

impl Map {
    pub fn new(path: &Path) -> Self {
        let db = Database::create(path).unwrap_or_else(|err| {
            panic!("Map::new: failed to create database at {:?}: {err}", path)
        });

        Self { db }
    }

    pub fn read(path: &Path) -> Self {
        if !path.exists() {
            panic!("Map::read: database file not found at {:?}", path);
        }

        if !path.is_file() {
            panic!("Map::read: path {:?} is not a file", path);
        }

        let db = Database::open(path).unwrap_or_else(|err| {
            panic!("Map::read: failed to open database at {:?}: {err}", path)
        });

        Self { db }
    }

    pub fn get_with_u128(&self, line_ref: u128) -> (u64, u64) {
        let read_transaction = self.db.begin_read().unwrap_or_else(|err| {
            panic!("Map::get_with_u128: failed to begin read transaction: {err}")
        });

        let table = read_transaction
            .open_table(MAP_TABLE)
            .unwrap_or_else(|err| panic!("Map::get_with_u128: failed to open map table: {err}"));

        match table.get(line_ref) {
            Ok(Some(value)) => value.value(),
            Ok(None) => panic!(
                "Map::get_with_u128: entry for hash {line_ref:#x} not found. Did you insert it?"
            ),
            Err(err) => {
                panic!("Map::get_with_u128: failed to fetch entry for hash {line_ref:#x}: {err}")
            },
        }
    }

    pub fn get_with_str(&self, line_ref: &str) -> (u64, u64) {
        self.get_with_u128(xxh3_128(line_ref.as_bytes()))
    }

    pub fn push_with_u128(&mut self, line_ref: u128, offset: u64, length: u64) {
        let write_transaction = self.db.begin_write().unwrap_or_else(|err| {
            panic!("Map::push_with_u128: failed to begin write transaction: {err}")
        });

        {
            let mut table = write_transaction
                .open_table(MAP_TABLE)
                .unwrap_or_else(|err| {
                    panic!("Map::push_with_u128: failed to open map table for writing: {err}")
                });

            table
                .insert(line_ref, (offset, length))
                .unwrap_or_else(|err| {
                    panic!(
                        "Map::push_with_u128: failed to insert entry for hash {line_ref:#x}: {err}"
                    )
                });
        }

        write_transaction.commit().unwrap_or_else(|err| {
            panic!("Map::push_with_u128: failed to commit transaction: {err}")
        });
    }

    pub fn push_with_str(&mut self, line_ref: &str, offset: u64, length: u64) {
        self.push_with_u128(xxh3_128(line_ref.as_bytes()), offset, length);
    }

    pub fn push_batch_with_str<'a, I>(&self, entries: I)
    where
        I: IntoIterator<Item = (&'a str, u64, u64)>,
    {
        let mut entries = entries.into_iter().peekable();

        if entries.peek().is_none() {
            return;
        }

        let write_transaction = self.db.begin_write().unwrap_or_else(|err| {
            panic!("Map::push_batch_with_str: failed to begin write transaction: {err}")
        });

        {
            let mut table = write_transaction
                .open_table(MAP_TABLE)
                .unwrap_or_else(|err| {
                    panic!("Map::push_batch_with_str: failed to open map table for writing: {err}")
                });

            for (line_ref, offset, length) in entries {
                let key = xxh3_128(line_ref.as_bytes());

                table.insert(key, (offset, length)).unwrap_or_else(|err| {
                    panic!(
                        "Map::push_batch_with_str: failed to insert entry for {line_ref:?}: {err}"
                    )
                });
            }
        }

        write_transaction.commit().unwrap_or_else(|err| {
            panic!("Map::push_batch_with_str: failed to commit transaction: {err}")
        });
    }
}
