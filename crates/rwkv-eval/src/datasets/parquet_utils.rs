use std::fs::File;
use std::path::Path;

use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};

pub fn read_parquet_items<T, P, F>(path: P, mut parse: F) -> Vec<T>
where
    P: AsRef<Path>,
    F: FnMut(&Row) -> T,
{
    let file = File::open(path.as_ref()).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();

    reader
        .get_row_iter(None)
        .unwrap()
        .map(|row| {
            let row = row.unwrap();
            parse(&row)
        })
        .collect()
}

pub fn get_string(row: &Row, name: &str) -> String {
    match get_field(row, name) {
        Field::Str(value) => value.clone(),
        _ => panic!("`{name}` should be string"),
    }
}

pub fn get_string_list(row: &Row, name: &str) -> Vec<String> {
    match get_field(row, name) {
        Field::ListInternal(list) => list
            .elements()
            .iter()
            .map(|field| match field {
                Field::Str(value) => value.clone(),
                _ => panic!("`{name}` should be list<string>"),
            })
            .collect(),
        _ => panic!("`{name}` should be list<string>"),
    }
}

pub fn get_u8(row: &Row, name: &str) -> u8 {
    match get_field(row, name) {
        Field::Byte(value) => u8::try_from(*value).unwrap(),
        Field::Short(value) => u8::try_from(*value).unwrap(),
        Field::Int(value) => u8::try_from(*value).unwrap(),
        Field::Long(value) => u8::try_from(*value).unwrap(),
        Field::UByte(value) => *value,
        Field::UShort(value) => u8::try_from(*value).unwrap(),
        Field::UInt(value) => u8::try_from(*value).unwrap(),
        Field::ULong(value) => u8::try_from(*value).unwrap(),
        _ => panic!("`{name}` should be integer"),
    }
}

fn get_field<'a>(row: &'a Row, name: &str) -> &'a Field {
    row.get_column_iter()
        .find(|(column_name, _)| column_name.as_str() == name)
        .map(|(_, field)| field)
        .unwrap()
}
