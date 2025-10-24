pub mod eval;
pub mod export;
pub mod infer;
pub mod model;
pub mod trace;
pub mod train;

#[macro_export]
macro_rules! fill_default {
    ($s:expr, $( $field:ident : $value:expr ),+ $(,)?) => {
        $(
            if $s.$field.is_none() {
                $s.$field = Some($value);
            }
        )+
    };
}
