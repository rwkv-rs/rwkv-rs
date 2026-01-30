use std::path::Path;

use bio::io::fasta;
use indicatif::{ProgressBar, ProgressStyle};
use rwkv_data::mmap::{bin::BinWriter, idx::IdxWriter, map::Map};
use rwkv_derive::LineRef;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, LineRef)]
pub struct AniMatItem {
    #[serde(rename = "Query_name")]
    #[line_ref]
    pub sequence_query_id: String,
    #[serde(rename = "Ref_name")]
    #[line_ref]
    pub sequence_ref_id: String,
    #[serde(rename = "ANI")]
    pub ani: f64,
}

#[tokio::main]
async fn main() {
    let mut bin = BinWriter::<u8>::new(
        Path::new("/public/home/ssjxzkz/Datasets/geno/ani/paired.bin"),
        1,
        4 * 1024 * 1024,
    );

    let map = Map::new(Path::new(
        "/public/home/ssjxzkz/Datasets/geno/ani/paired.map",
    ));

    let mut idx = IdxWriter::<AniMatItem>::new(
        Path::new("/public/home/ssjxzkz/Datasets/geno/ani/paired.idx"),
        4 * 1024 * 1024,
    );

    let fasta_reader = fasta::Reader::from_file_with_capacity(
        4 * 1024 * 1024,
        "/public/home/ssjxzkz/Datasets/geno/ani/paired.fasta",
    )
    .unwrap();

    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(Path::new(
            "/public/home/ssjxzkz/Datasets/geno/ani/output.ani_matrix.tsv",
        ))
        .unwrap();

    // Create progress bar for FASTA processing
    let fasta_pb = ProgressBar::new_spinner();

    fasta_pb.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"])
            .template("{spinner:.green} [{elapsed}] FASTA: {pos} records ({per_sec})")
            .unwrap(),
    );

    fasta_pb.set_message("Processing FASTA records");

    let mut tokens = Vec::with_capacity(4096);

    let mut map_entries = Vec::with_capacity(16_384);

    for record in fasta_reader.records() {
        let record = record.unwrap();

        let id = record.id().to_owned();

        let seq = record.seq();

        tokens.clear();

        tokens.reserve(seq.len());

        tokens.extend(seq.iter().map(|bp| match bp {
            b'A' | b'a' => 0u8,
            b'C' | b'c' => 1u8,
            b'G' | b'g' => 2u8,
            b'T' | b't' | b'U' | b'u' => 3u8,
            _ => 4u8,
        }));

        let (offset, length) = bin.push(&tokens);

        map_entries.push((id, offset, length));

        fasta_pb.inc(1);
    }

    fasta_pb.finish_with_message("FASTA processing completed");

    map.push_batch_with_str(
        map_entries
            .iter()
            .map(|(id, offset, length)| (id.as_str(), *offset, *length)),
    );

    // Create progress bar for CSV processing
    let csv_pb = ProgressBar::new_spinner();

    csv_pb.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"])
            .template("{spinner:.blue} [{elapsed}] CSV: {pos} records ({per_sec})")
            .unwrap(),
    );

    csv_pb.set_message("Processing CSV records");

    for record in csv_reader.deserialize::<AniMatItem>() {
        let record: AniMatItem = record.unwrap();

        idx.push(&record, &map);

        csv_pb.inc(1);
    }

    csv_pb.finish_with_message("CSV processing completed");

    bin.update_metadata();

    idx.update_metadata();
}
