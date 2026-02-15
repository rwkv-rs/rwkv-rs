//! # RWKV 派生宏库
//!
//! 这个库提供了两个主要的过程宏，用于简化 RWKV 项目中常见模式的代码生成：
//! - `LineRef`: 为包含行引用字段的结构体生成序列化/反序列化实现
//! - `ConfigBuilder`: 为配置结构体生成建造者模式的构建器
//!
//! ## 使用示例
//!
//! ### LineRef 宏
//! ```rust,ignore
//! use rwkv_derive::LineRef;
//!
//! #[derive(LineRef)]
//! struct MyStruct {
//!     #[line_ref]
//!     name: String,
//!     value: u32,
//! }
//! ```
//!
//! ### ConfigBuilder 宏
//! ```rust,ignore
//! use rwkv_derive::ConfigBuilder;
//!
//! #[derive(ConfigBuilder)]
//! #[config_builder(raw = "MyConfigRaw", cell = "MY_CONFIG")]
//! struct MyConfig {
//!     host: String,
//!     #[config_builder(skip_raw)]
//!     port: Option<u16>,
//! }
//! ```

// 导入过程宏相关的基础库
use proc_macro::TokenStream; // 过程宏的核心类型，用于接收和返回TokenStream
use quote::quote; // 用于生成Rust代码的引用宏库
use syn::{
    // syn库用于解析Rust语法树，提供以下类型：
    Attribute,         // 表示属性，如 `#[derive(Debug)]` 或 `#[line_ref]`
    Data,              // 表示数据结构（Struct、Enum、Union）
    DeriveInput,       // 表示派生宏的输入，即整个结构体/枚举定义
    Fields,            // 表示结构体字段集合
    GenericArgument,   // 表示泛型参数，如 `Vec<T>` 中的 `T`
    PathArguments,     // 表示路径参数，如 `Vec<String>` 中的 `String`
    Type,              // 表示类型，如 `String`、`u32`、`Option<T>` 等
    parse_macro_input, // 宏，用于解析输入TokenStream为指定的类型
};

/// ## LineRef 派生宏
///
/// `LineRef` 是一个过程宏，用于为结构体生成内存映射序列化支持。
/// 它主要用于处理包含行引用字段的结构体，这些字段通常是 `String` 或 `u128`
/// 类型， 用于表示在内存映射文件中的引用位置（偏移量和长度）。
///
/// ### 生成的代码结构
/// 对于一个带有 `#[line_ref]` 属性的结构体，此宏会生成：
/// 1. 一个序列化结构体 `{StructName}Serialized`，包含所有字段的序列化版本
/// 2. `serialized_size()` 方法，返回序列化结构体的大小
/// 3. `LineRefSample` trait 的实现，提供序列化和反序列化功能
///
/// ### 支持的 line_ref 类型
/// - `String`: 字符串类型，会被序列化为 UTF-8 字节序列
/// - `u128`: 128位无符号整数，直接转换为字节序列
///
/// ### 使用限制
/// - 只能用于命名结构体（struct with named fields）
/// - 不支持枚举或元组结构体
/// - line_ref 字段必须是 String 或 u128 类型
///
/// ## 参数说明
/// - `input`: 输入的 TokenStream，包含完整的结构体定义
/// - 返回：生成的代码的 TokenStream
#[proc_macro_derive(LineRef, attributes(line_ref))]
pub fn derive_line_ref(input: TokenStream) -> TokenStream {
    // 第一步：解析输入的结构体定义
    let input = parse_macro_input!(input as DeriveInput); // 将TokenStream解析为DeriveInput结构体
    let name = &input.ident; // 获取结构体名称，如 "MyStruct"
    // 第二步：验证输入是结构体且具有命名字段
    let Data::Struct(data_struct) = &input.data else {
        panic!("LineRef can only be derived for structs"); // 确保只能用于结构体
    };

    let Fields::Named(fields) = &data_struct.fields else {
        panic!("LineRef can only be derived for structs with named fields"); // 确保是命名字段结构体
    };

    // 第三步：分类处理结构体字段
    let mut line_ref_fields = Vec::new(); // 存储带有 line_ref 属性的字段及其类型信息
    let mut regular_fields = Vec::new(); // 存储常规字段（非 line_ref 字段）
    // 遍历所有字段，根据是否有 line_ref 属性进行分类
    for field in &fields.named {
        let field_name = field.ident.as_ref().unwrap(); // 获取字段名称，如 "name"、"value"
        let field_type = &field.ty; // 获取字段类型，如 "String"、"u32"
        // 检查字段是否有 line_ref 属性
        if field
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("line_ref"))
        {
            // 如果有 line_ref 属性，提取其类型信息（String 或 u128）
            let line_ref_type = extract_line_ref_type(field_type);

            line_ref_fields.push((field_name, line_ref_type));
        } else {
            // 否则作为常规字段处理
            regular_fields.push((field_name, field_type));
        }
    }

    // 第四步：准备序列化字段的生成
    // 创建所有字段名称的有序列表，用于确保序列化字段的顺序一致性
    let mut all_field_names: Vec<_> = line_ref_fields
        .iter()
        .map(
            |(name, _)| name.to_string(), // 提取 line_ref 字段名称
        )
        .collect();

    all_field_names.extend(regular_fields.iter().map(|(name, _)| name.to_string())); // 添加常规字段名称
    all_field_names.sort(); // 排序确保字段顺序一致
    // 用于生成序列化相关代码的向量
    let mut serialized_fields = Vec::new(); // 序列化结构体的字段定义
    let mut to_serialized_lets = Vec::new(); // 序列化时的变量绑定代码
    let mut to_serialized_fields = Vec::new(); // 序列化结构体实例化时的字段赋值
    let mut from_serialized_assignments = Vec::new(); // 反序列化时的字段赋值代码
    // 第五步：为每个字段生成序列化相关代码
    // 按照排序后的字段名称顺序处理每个字段
    for field_name_str in &all_field_names {
        // 查找当前字段是否是 line_ref 字段
        if let Some((field_name, line_ref_type)) = line_ref_fields
            .iter()
            .find(|(name, _)| name.to_string() == *field_name_str)
        {
            // 处理 line_ref 字段：生成偏移量和长度字段
            let offset_field = quote::format_ident!("{}_offset", field_name); // 如 "name_offset"
            let length_field = quote::format_ident!("{}_length", field_name); // 如 "name_length"
            // 为序列化结构体添加偏移量和长度字段（都是 u64 类型）
            serialized_fields.push(quote! { #offset_field: u64 });

            serialized_fields.push(quote! { #length_field: u64 });

            // 根据 line_ref 字段的类型生成不同的序列化/反序列化代码
            match line_ref_type {
                LineRefType::String => {
                    // 处理 String 类型：从内存映射中获取字符串的位置信息
                    to_serialized_lets.push(quote! {
                        let (#offset_field, #length_field) = map.get_with_str(&self.#field_name);
                    });

                    // 反序列化：从二进制数据中读取 UTF-8 字符串
                    from_serialized_assignments.push(quote! {
                        #field_name: {
                            let tokens = bin.get(data.#offset_field, data.#length_field);
                            String::from_utf8(tokens.into_owned()).unwrap()
                        }
                    });
                }
                LineRefType::U128 => {
                    // 处理 u128 类型：从内存映射中获取 u128 的位置信息
                    to_serialized_lets.push(quote! {
                        let (#offset_field, #length_field) = map.get_with_u128(self.#field_name);
                    });

                    // 反序列化：从二进制数据中读取小端序的 u128 值
                    from_serialized_assignments.push(quote! {
                        #field_name: {
                            let tokens = bin.get(data.#offset_field, data.#length_field);
                            let slice = tokens.as_ref();
                            let bytes: [u8; 16] = slice.try_into().unwrap();
                            u128::from_le_bytes(bytes)
                        }
                    });
                }
            }

            // 为序列化结构体实例化添加偏移量和长度字段
            to_serialized_fields.push(quote! { #offset_field });

            to_serialized_fields.push(quote! { #length_field });

        // 处理常规字段（非 line_ref 字段）
        } else if let Some((field_name, field_type)) = regular_fields
            .iter()
            .find(|(name, _)| name.to_string() == *field_name_str)
        {
            // 为序列化结构体添加常规字段
            serialized_fields.push(quote! { #field_name: #field_type });

            // 反序列化时直接从序列化数据中读取
            from_serialized_assignments.push(quote! { #field_name: data.#field_name });

            // 序列化时直接使用原字段值
            to_serialized_fields.push(quote! { #field_name: self.#field_name });
        } else {
            // 理论上不会出现这种情况，但为了调试保留错误处理
            panic!(
                "Field {} not found in either line_ref_fields or regular_fields",
                field_name_str
            );
        }
    }

    // 第六步：生成最终的代码
    let serialized_name = quote::format_ident!("{}Serialized", name); // 生成序列化结构体的名称，如 "MyStructSerialized"
    // 使用 quote! 宏生成最终的 Rust 代码
    let expanded = quote! {
        // 定义序列化结构体，具有 C 语言兼容的内存布局
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        #[repr(C)]  // 确保内存布局与 C 兼容，用于内存映射序列化
        pub struct #serialized_name {
            #(#serialized_fields,)*  // 包含所有字段：常规字段 + line_ref 字段的偏移量/长度
        }

        // 为原始结构体实现序列化大小计算方法
        impl #name {
            /// 返回序列化结构体的大小（字节数）
            /// 用于预先分配内存或计算存储需求
            pub fn serialized_size() -> usize {
                std::mem::size_of::<#serialized_name>()
            }
        }

        // 为原始结构体实现 LineRefSample trait，提供序列化/反序列化功能
        impl rwkv_data::mmap::idx::LineRefSample for #name {
            type Serialized = #serialized_name;  // 指定关联类型为生成的序列化结构体

            /// 将结构体序列化为内存映射友好的格式
            /// - `map`: 内存映射对象，用于获取字符串/数据的引用位置
            /// - 返回：序列化结构体，包含偏移量、长度和常规字段数据
            fn to_serialized(&self, map: &rwkv_data::mmap::map::Map) -> Self::Serialized {
                #(#to_serialized_lets)*  // 执行所有变量绑定，获取偏移量和长度

                // 创建序列化结构体实例
                #serialized_name {
                    #(#to_serialized_fields,)*  // 填充所有字段的值
                }
            }

            /// 从序列化数据反序列化回原始结构体
            /// - `data`: 序列化的数据结构
            /// - `bin`: 二进制数据源，用于读取字符串和 u128 数据
            /// - 返回：反序列化后的原始结构体实例
            fn from_serialized(data: &Self::Serialized, bin: &rwkv_data::mmap::bin::BinReader<u8>) -> Self {
                Self {
                    #(#from_serialized_assignments,)*  // 为所有字段赋值，包括从二进制数据读取的 line_ref 字段
                }
            }
        }
    };

    // 返回生成的代码作为 TokenStream
    TokenStream::from(expanded)
}

/// 表示支持的 line_ref 字段类型枚举
/// 用于标识字段应该如何进行内存映射序列化处理
#[derive(Debug)]
enum LineRefType {
    String, // 字符串类型，需要 UTF-8 编码/解码
    U128,   // 128位无符号整数，直接字节序转换
}

/// 从类型路径中提取 line_ref 字段的具体类型
/// 用于确定字段应该使用哪种序列化/反序列化策略
///
/// ## 参数
/// - `ty`: 要分析的类型引用
///
/// ## 返回
/// 返回对应的 `LineRefType` 枚举值
///
/// ## 错误处理
/// 当遇到不支持的类型时会触发 panic，目前仅支持 String 和 u128
fn extract_line_ref_type(ty: &Type) -> LineRefType {
    match ty {
        // 处理路径类型，如 `String`、`u128`、`std::string::String` 等
        Type::Path(type_path) => {
            // 获取路径的最后一个片段，即类型名称
            let last_segment = type_path.path.segments.last().unwrap();

            match last_segment.ident.to_string().as_str() {
                "String" => LineRefType::String, // 字符串类型
                "u128" => LineRefType::U128,     // 128位无符号整数类型
                _ => panic!(
                    "Unsupported line_ref type. Only String and u128 are supported. Got: {}",
                    last_segment.ident
                ),
            }
        }
        // 对于其他类型形式，暂时不支持
        _ => panic!(
            "Unsupported line_ref type. Only String and u128 are supported. Expected a simple \
             path type."
        ),
    }
}

/// ## ConfigBuilder 派生宏
///
/// `ConfigBuilder` 是一个过程宏，用于为结构体生成建造者模式的配置构建器。
/// 它简化了配置对象的创建过程，支持可选字段、类型转换和全局单例管理。
///
/// ### 生成的代码结构
/// 对于一个带有 `#[config_builder]` 属性的结构体，此宏会生成：
/// 1. 一个建造者结构体 `{StructName}Builder`，包含所有字段的 Option 版本
/// 2. `new()` 方法，创建空的建造者实例
/// 3. `set_*()` 方法链，用于设置各个字段的值
/// 4. `get_*()` 方法，用于获取当前设置的值
/// 5. `load_from_raw()` 方法，从原始配置加载值
/// 6. `build()` 方法，验证并构建最终配置，使用 Arc 单例存储
///
/// ### 支持的属性
/// - `config_builder(raw = "RawType", cell = "CELL_NAME")`:
///   指定原始类型和全局单元名称
/// - `skip_raw`: 标记字段不从原始配置加载
///
/// ### 类型处理规则
/// - 非 Option<T> 字段在 Builder 中变为 Option<T>，build 时必须提供值
/// - Option<T> 字段在 Builder 中保持 Option<T>，build 时可为 None
///
/// ### 使用限制
/// - 只能用于命名结构体（struct with named fields）
/// - 不支持枚举或元组结构体
/// - 结构体必须有 `#[config_builder(...)]` 属性
///
/// ## 参数说明
/// - `input`: 输入的 TokenStream，包含完整的结构体定义
/// - 返回：生成的代码的 TokenStream
#[proc_macro_derive(ConfigBuilder, attributes(config_builder, skip_raw))]
pub fn derive_config_builder(input: TokenStream) -> TokenStream {
    // 第一步：解析输入的结构体定义
    let input = parse_macro_input!(input as DeriveInput); // 将TokenStream解析为DeriveInput结构体
    let name = &input.ident; // 获取结构体名称，如 "MyConfig"
    // 第二步：验证输入是结构体且具有命名字段
    let Data::Struct(data_struct) = &input.data else {
        panic!("ConfigBuilder can only be derived for structs"); // 确保只能用于结构体
    };

    let Fields::Named(fields) = &data_struct.fields else {
        panic!("ConfigBuilder can only be derived for structs with named fields"); // 确保是命名字段结构体
    };

    // 第三步：解析结构体级别的 config_builder 属性
    let config_attrs = parse_config_builder_attrs(&input.attrs); // 解析 raw 和 cell 参数
    // 将字符串形式的类型路径转换为 syn::Path 对象
    let raw_type: syn::Path =
        syn::parse_str(&config_attrs.raw_type).expect("Invalid raw type path");

    let cell_name: syn::Path = syn::parse_str(&config_attrs.cell_name).expect("Invalid cell name");

    // 生成建造者结构体的名称，如 "MyConfigBuilder"
    let builder_name = quote::format_ident!("{}Builder", name);

    // 第四步：生成 Builder 结构体的字段定义
    // 为每个原始字段生成对应的 Builder 字段，统一包装为 Option<T>
    let builder_fields: Vec<_> = fields
        .named
        .iter()
        .map(|field| {
            let field_name = &field.ident; // 字段名称，如 "host"、"port"
            let field_type = &field.ty; // 字段类型，如 "String"、"Option<u16>"
            // 根据原始字段类型确定 Builder 中对应的类型：
            // - 如果已经是 Option<T>，则保持不变
            // - 如果是 T，则包装成 Option<T>
            let builder_field_type = if is_option_type(field_type) {
                quote! { #field_type } // 已经是 Option<T>，直接使用
            } else {
                quote! { Option<#field_type> } // 包装成 Option<T>
            };

            // 生成字段定义代码
            quote! {
                #field_name: #builder_field_type
            }
        })
        .collect();

    // 第五步：生成设置方法（set_* 方法）
    // 这些方法用于链式调用，设置各个字段的值
    let set_methods: Vec<_> = fields
        .named
        .iter()
        .map(|field| {
            let field_name = &field.ident;

            let field_type = &field.ty;

            // 生成方法名称，如 "set_host"、"set_port"
            let method_name = quote::format_ident!("set_{}", field_name.as_ref().unwrap());

            // 确定方法参数类型：始终接受 Option<T> 形式的参数
            // 这样可以灵活设置值或显式设置为 None
            let builder_field_type = if is_option_type(field_type) {
                quote! { #field_type } // 原始类型已经是 Option<T>，参数也是 Option<T>
            } else {
                quote! { Option<#field_type> } // 原始类型是 T，参数是 Option<T>
            };

            // 生成设置方法的代码：链式调用，返回 &mut self
            quote! {
                /// 设置字段 `#field_name` 的值，支持链式调用
                /// 参数为 Option<T>，允许设置为 None
                pub fn #method_name(&mut self, val: #builder_field_type) -> &mut Self {
                    self.#field_name = val;  // 设置字段值
                    self  // 返回 self 以支持链式调用
                }
            }
        })
        .collect();

    // 第六步：生成获取方法（get_* 方法）
    // 这些方法用于获取当前 Builder 中字段的设置值
    let get_methods: Vec<_> = fields
        .named
        .iter()
        .map(|field| {
            let field_name = &field.ident;

            let field_type = &field.ty;

            // 生成方法名称，如 "get_host"、"get_port"
            let method_name = quote::format_ident!("get_{}", field_name.as_ref().unwrap());

            // 确定方法的返回类型：始终返回 Option<T>
            // - 如果原始类型是 Option<T>，返回 Option<U>（内部类型）
            // - 如果原始类型是 T，返回 Option<T>
            let return_type = if is_option_type(field_type) {
                let inner_type = extract_option_inner_type(field_type);

                quote! { Option<#inner_type> } // 返回 Option<内部类型>
            } else {
                quote! { Option<#field_type> } // 返回 Option<T>
            };

            // 生成获取方法的代码
            quote! {
                /// 获取字段 `#field_name` 的当前设置值
                /// 始终返回 Option<T>，表示可能未设置
                pub fn #method_name(&self) -> #return_type {
                    self.#field_name.clone()  // 返回克隆的 Option 值
                }
            }
        })
        .collect();

    // 第七步：生成 build 方法中的字段验证和赋值代码
    // 为最终配置结构体生成字段赋值表达式，包含必要的验证逻辑
    let build_fields: Vec<_> = fields.named.iter().map(|field| {
        let field_name = &field.ident; // 字段名称，如 "host"、"port"
        let field_type = &field.ty;    // 字段类型，如 "String"、"Option<u16>"

        if is_option_type(field_type) {
            // 如果原始字段是 Option<T>，直接传递 Builder 中的 Option<T>
            // 不需要验证，因为 Option<T> 本身就可以是 None
            quote! {
                #field_name: self.#field_name  // 直接使用 Builder 中的值
            }
        } else {
            // 如果原始字段是 T，需要从 Builder 的 Option<T> 中提取值
            // 必须验证非空，否则构建失败
            let field_name_str = field_name.as_ref().unwrap().to_string();
            quote! {
                // 尝试提取必填字段的值，如果为 None 则报错
                #field_name: self.#field_name.expect(&format!("missing field: {}", #field_name_str))
            }
        }
    }).collect();

    // 第八步：生成 load_from_raw 方法的字段映射代码
    // 支持字段级别的 skip_raw 属性，用于跳过从原始配置加载某些字段
    let load_from_raw_calls: Vec<_> = fields.named.iter().map(|field| {
        let field_name = &field.ident;
        // 生成对应的 set 方法名称，如 "set_host"、"set_port"
        let method_name = quote::format_ident!("set_{}", field_name.as_ref().unwrap());

        // 检查字段是否有 skip_raw 属性
        let has_skip_raw = field.attrs.iter().any(|attr| attr.path().is_ident("skip_raw"));

        if has_skip_raw {
            // 如果有 skip_raw 属性，该字段不从原始配置加载，显式设置为 None
            quote! {
                .#method_name(None)  // 跳过此字段，不从 raw 中加载
            }
        } else {
            // 如果没有 skip_raw 属性，从原始配置中加载值
            // 使用 IntoBuilderOption trait 进行类型转换
            quote! {
                .#method_name(crate::config_builder_helpers::IntoBuilderOption::into_builder_option(raw.#field_name))
            }
        }
    }).collect();

    // 第九步：生成最终的代码
    // 使用 quote! 宏生成完整的建造者结构体和实现代码
    let expanded = quote! {
        /// 建造者结构体，包含所有配置字段的 Option 版本
        /// 用于逐步构建配置对象，支持链式调用
        #[derive(Default)]
        pub struct #builder_name {
            #(#builder_fields,)*  // 所有字段的 Option 版本
        }

        // 为建造者结构体实现各种方法
        impl #builder_name {
            /// 创建一个新的空建造者实例
            /// 所有字段初始为 None，需要通过 set_* 方法设置
            pub fn new() -> Self {
                Self::default()  // 使用 Default trait 创建空实例
            }

            // 包含所有生成的 set_* 方法
            #(#set_methods)*

            // 包含所有生成的 get_* 方法
            #(#get_methods)*

            /// 从原始配置类型加载初始值
            /// - `raw`: 原始配置实例，包含默认或外部配置值
            /// - 返回：填充了初始值的建造者实例
            ///
            /// 注意：带有 `skip_raw` 属性的字段将被设置为 None
            pub fn load_from_raw(raw: #raw_type) -> Self {
                let mut builder = Self::new();  // 创建空建造者
                builder
                    #(#load_from_raw_calls)*;  // 链式调用所有 set 方法加载值
                builder
            }

            /// 构建最终的配置对象并存储为全局单例
            /// - 执行字段验证：必填字段（非 Option<T>）必须已设置
            /// - 创建配置结构体实例
            /// - 包装为 Arc（原子引用计数）以支持多所有权
            /// - 存储到指定的全局单元中，确保单例模式
            /// - 返回 Arc 包装的配置对象
            ///
            /// ## 错误处理
            /// - 如果必填字段未设置，会触发 panic 并显示缺失字段名称
            /// - 如果全局单元已被初始化，也会触发 panic
            pub fn build(self) -> std::sync::Arc<#name> {
                // 构建配置结构体，验证必填字段
                let config = #name {
                    #(#build_fields,)*  // 为所有字段赋值，包含验证逻辑
                };

                // 包装为 Arc 以支持共享所有权
                let arc_config = std::sync::Arc::new(config);

                // 存储到全局单元，确保单例模式
                // 如果单元已被初始化，会返回错误并触发 panic
                #cell_name.set(arc_config.clone()).expect(&format!("{} already initialized", stringify!(#cell_name)));

                arc_config  // 返回配置对象的 Arc 引用
            }
        }
    };

    // 返回生成的代码作为 TokenStream
    TokenStream::from(expanded)
}

/// 表示从 `config_builder` 属性解析出的配置信息
/// 包含原始配置类型路径和全局单元名称
#[derive(Debug)]
struct ConfigBuilderAttrs {
    raw_type: String,  // 原始配置类型的字符串路径，如 "crate::config::MyConfigRaw"
    cell_name: String, // 全局单元名称，如 "MY_CONFIG"
}

/// 解析结构体上的 `config_builder` 属性，提取必要参数
/// 用于确定原始配置类型和全局单例存储单元的名称
///
/// ## 参数
/// - `attrs`: 结构体上的所有属性列表
///
/// ## 返回
/// 返回解析出的配置信息结构体
///
/// ## 解析逻辑
/// - 查找 `config_builder` 属性
/// - 提取 `raw = "..."` 参数作为原始配置类型路径
/// - 提取 `cell = "..."` 参数作为全局单元名称
/// - 验证必需参数都已提供，否则触发 panic
///
/// ## 错误处理
/// - 如果缺少 `raw` 参数，会触发 panic 并提示错误
/// - 如果缺少 `cell` 参数，会触发 panic 并提示错误
fn parse_config_builder_attrs(attrs: &[Attribute]) -> ConfigBuilderAttrs {
    // 初始化参数变量
    let mut raw_type = None; // 原始配置类型路径
    let mut cell_name = None; // 全局单元名称
    // 遍历所有属性，寻找 config_builder 属性
    for attr in attrs {
        if attr.path().is_ident("config_builder") {
            // 将属性转换为字符串形式，清理换行符便于解析
            let attr_str = quote!(#attr)
                .to_string()
                .replace('\n', " ")
                .replace('\r', "");

            // 解析 raw = "..." 参数
            if let Some(raw_start) = attr_str.find("raw = \"") {
                let raw_start = raw_start + 7; // 跳过 'raw = "' 的长度
                if let Some(raw_end) = attr_str[raw_start..].find('"') {
                    // 提取引号内的内容作为原始类型路径
                    raw_type = Some(attr_str[raw_start..raw_start + raw_end].to_string());
                }
            }

            // 解析 cell = "..." 参数
            if let Some(cell_start) = attr_str.find("cell = \"") {
                let cell_start = cell_start + 8; // 跳过 'cell = "' 的长度
                if let Some(cell_end) = attr_str[cell_start..].find('"') {
                    // 提取引号内的内容作为单元名称
                    cell_name = Some(attr_str[cell_start..cell_start + cell_end].to_string());
                }
            }
        }
    }

    // 构建并返回配置信息，确保必需参数都已提供
    ConfigBuilderAttrs {
        raw_type: raw_type.expect("config_builder attribute must specify 'raw' parameter"),
        cell_name: cell_name.expect("config_builder attribute must specify 'cell' parameter"),
    }
}

/// 判断类型是否为 Option<T> 类型
/// 用于在代码生成时区别处理必填字段和可选字段
///
/// ## 参数
/// - `ty`: 要检查的类型引用
///
/// ## 返回
/// 如果类型路径以 "Option" 结尾，返回 true；否则返回 false
///
/// ## 示例
/// ```ignore
/// assert_eq!(is_option_type(&syn::parse_str::<Type>("Option<String>").unwrap()), true);
/// assert_eq!(is_option_type(&syn::parse_str::<Type>("String").unwrap()), false);
/// ```
fn is_option_type(ty: &Type) -> bool {
    // 检查类型路径的最后一个片段是否为 "Option"
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident == "Option";
    }

    false // 不是路径类型或最后一个片段不是 Option
}

/// 提取 Option<T> 类型中的内部类型 T
/// 用于在生成 get_* 方法时确定正确的返回类型
///
/// ## 参数
/// - `ty`: Option<T> 类型的引用，必须确保调用前已验证为 Option 类型
///
/// ## 返回
/// 返回内部类型 T 的 TokenStream 表示
///
/// ## 泛型参数处理
/// - 解析 `Option<T>` 中的泛型参数 `T`
/// - 如果解析失败，返回空的元组类型 `()` 作为后备
///
/// ## 示例
/// ```ignore
/// let inner_type = extract_option_inner_type(&syn::parse_str::<Type>("Option<String>").unwrap());
/// // inner_type 将是 "String" 的 TokenStream 表示
/// ```
fn extract_option_inner_type(ty: &Type) -> proc_macro2::TokenStream {
    // 解析 Option<T> 类型，提取内部类型 T
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
        && segment.ident == "Option"  // 确认是 Option 类型
        && let PathArguments::AngleBracketed(args) = &segment.arguments  // 获取尖括号内的泛型参数
        && let Some(GenericArgument::Type(inner_type)) = args.args.first()
    // 提取第一个类型参数
    {
        return quote! { #inner_type }; // 返回内部类型
    }

    quote! { () } // 后备：如果解析失败，返回空元组类型
}
