pub mod bin;
pub mod bin_old;
pub mod dtype;
pub mod idx;
pub mod map;
pub mod sample;


const MMAP_VERSION: [u8; 8] = 1u64.to_le_bytes();
