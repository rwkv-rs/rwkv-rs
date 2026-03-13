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

pub fn get_optional_string(row: &Row, name: &str) -> Option<String> {
    match get_field(row, name) {
        Field::Null => None,
        Field::Str(value) => Some(value.clone()),
        _ => panic!("`{name}` should be string or null"),
    }
}

pub fn get_string_list(row: &Row, name: &str) -> Vec<String> {
    match get_field(row, name) {
        Field::ListInternal(list) => get_string_list_from_fields(list.elements(), name),
        _ => panic!("`{name}` should be list<string>"),
    }
}

pub fn get_optional_string_list(row: &Row, name: &str) -> Option<Vec<String>> {
    match get_field(row, name) {
        Field::Null => None,
        Field::ListInternal(list) => Some(get_string_list_from_fields(list.elements(), name)),
        _ => panic!("`{name}` should be list<string> or null"),
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

pub fn get_i64(row: &Row, name: &str) -> i64 {
    match get_field(row, name) {
        Field::Byte(value) => i64::from(*value),
        Field::Short(value) => i64::from(*value),
        Field::Int(value) => i64::from(*value),
        Field::Long(value) => *value,
        Field::UByte(value) => i64::from(*value),
        Field::UShort(value) => i64::from(*value),
        Field::UInt(value) => i64::from(*value),
        Field::ULong(value) => i64::try_from(*value).unwrap(),
        _ => panic!("`{name}` should be integer"),
    }
}

fn get_field<'a>(row: &'a Row, name: &str) -> &'a Field {
    row.get_column_iter()
        .find(|(column_name, _)| column_name.as_str() == name)
        .map(|(_, field)| field)
        .unwrap()
}

fn get_string_list_from_fields(fields: &[Field], name: &str) -> Vec<String> {
    fields
        .iter()
        .map(|field| match field {
            Field::Str(value) => value.clone(),
            _ => panic!("`{name}` should be list<string>"),
        })
        .collect()
}
