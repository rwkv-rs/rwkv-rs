use std::fs::File;

use burn::{
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    module::Module,
    prelude::{Backend, Shape, TensorData},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Float, Int, Tensor, cast::ToElement},
};
use itertools::izip;
use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;

use crate::auto_regressive_model::{AutoRegressiveModel, AutoRegressiveModelConfig};

/// 全局后端类型定义

pub type TestBackend = Cuda<f32, i32>;

pub type TestAutodiffBackend = Autodiff<TestBackend>;

pub const MIN_PASS_RATE: f64 = 0.99;

pub const RELATIVE_ERROR: f64 = 0.01;

pub const TEST_BATCH_SIZE: usize = 1;

pub const TEST_CONTEXT_LENGTH: usize = 128;

pub const TEST_EMBEDDED_DIM: usize = 768;

pub const TEST_NUM_HEADS: usize = 12;

pub const TEST_HEAD_SIZE: usize = 64;

/// 获取全局设备

pub fn get_global_device() -> CudaDevice {
    CudaDevice::default()
}

/// 获取全局模型（每次创建新实例，避免Sync问题）

pub fn get_global_model() -> AutoRegressiveModel<TestAutodiffBackend> {
    let device = get_global_device();

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
        .load(
            "../../weights/rwkv7-g1-0.1b-20250307-ctx4096.mpk".into(),
            &device,
        )
        .expect("Failed to load weights for unit tests");

    AutoRegressiveModelConfig::new(12, 65536, 768, 12, 64)
        .init::<TestAutodiffBackend, u16>(&device)
        .load_record(record)
}

pub fn assert_closeness<B: Backend, const D: usize>(
    actual: &Tensor<B, D>,
    expected: &Tensor<B, D>,
    module_name: &str,
    min_pass_rate: f64,
    max_relative_error: f64,
) {
    assert_eq!(
        actual.shape().dims,
        expected.shape().dims,
        "🚨 DIMENSION MISMATCH: Layer '{}' shape mismatch. Actual: {:?}, Expected: {:?}",
        module_name,
        actual.shape().dims,
        expected.shape().dims
    );

    let pass_rate = get_pass_rate(actual, expected, max_relative_error);

    assert!(
        pass_rate >= min_pass_rate,
        "🚨 UNIT TEST FAILURE: Layer '{}' precision check failed!\n\
         📋 Required: {:.1}% pass rate within {:.1}% relative error\n\
         📋 Actual:   {:.1}% pass rate\n\
         📋 Gap:      {:.1}% below threshold",
        module_name,
        min_pass_rate * 100.0,
        max_relative_error * 100.0,
        pass_rate * 100.0,
        (min_pass_rate - pass_rate) * 100.0
    );

    println!("  ✅ [PASS] Unit test passed for {}\n", module_name);
}

pub fn assert_closeness_multi<B: Backend, const D: usize>(
    actual_vec: Vec<Tensor<B, D>>,
    expected_vec: Vec<Tensor<B, D>>,
    module_name_vec: Vec<String>,
    min_pass_rate: f64,
    max_relative_error: f64,
) {
    let mut pass_rate_vec = vec![];

    let mut is_pass_vec = vec![];

    for (actual, expected) in izip!(actual_vec, expected_vec) {
        let pass_rate = get_pass_rate(&actual, &expected, max_relative_error);

        pass_rate_vec.push(pass_rate);

        is_pass_vec.push(pass_rate >= min_pass_rate);
    }

    for (module_name, pass_rate, is_pass) in
        izip!(module_name_vec, pass_rate_vec, is_pass_vec.clone())
    {
        if !is_pass {
            eprintln!(
                "🚨 UNIT TEST FAILURE: Layer '{}' precision check failed!\n\
                 📋 Required: {:.3}% pass rate within {:.1}% relative error\n\
                 📋 Actual:   {:.3}% pass rate\n\
                 📋 Gap:      {:.3}% below threshold\n",
                module_name,
                min_pass_rate * 100.0,
                max_relative_error * 100.0,
                pass_rate * 100.0,
                (min_pass_rate - pass_rate) * 100.0
            )
        } else {
            println!("  ✅ [PASS] Unit test passed for {}\n", module_name);
        }
    }

    assert!(
        is_pass_vec.iter().all(|&is_pass| is_pass),
        "UNIT TEST FAILURE!"
    );
}

fn get_pass_rate<B: Backend, const D: usize>(
    actual: &Tensor<B, D>,
    expected: &Tensor<B, D>,
    max_relative_error: f64,
) -> f64 {
    let absolute_tolerance = 1e-5;

    let absolute_error = actual.clone().sub(expected.clone()).abs();

    let expected_abs = expected.clone().abs();

    let tolerance = expected_abs * max_relative_error + absolute_tolerance;

    let pass_mask = absolute_error.lower_equal(tolerance);

    let pass_count = pass_mask.float().sum().into_scalar().to_f64();

    let total_elements = actual.shape().num_elements() as f64;

    pass_count / total_elements
}

pub fn load_expected_f32<const D: usize>(file_name: &str) -> Tensor<TestAutodiffBackend, D, Float> {
    let file = File::open(format!("../../data/variable_tracking/{}.npy", file_name)).unwrap();

    let array: ArrayD<f32> = ArrayD::<f32>::read_npy(file).unwrap();

    let device = get_global_device();

    array_npy2burn_f32::<TestAutodiffBackend, D>(&array, &device)
}

pub fn load_expected_i64<const D: usize>(file_name: &str) -> Tensor<TestAutodiffBackend, D, Int> {
    let file = File::open(format!("../../data/variable_tracking/{}.npy", file_name)).unwrap();

    let array: ArrayD<i64> = ArrayD::<i64>::read_npy(file).unwrap();

    let device = get_global_device();

    array_npy2burn_i64::<TestAutodiffBackend, D>(&array, &device)
}

pub fn array_npy2burn_f32<B: Backend, const D: usize>(
    array: &ArrayD<f32>,
    device: &B::Device,
) -> Tensor<B, D, Float> {
    let shape = array.shape().to_vec();

    let data: Vec<f32> = array.iter().cloned().collect();

    let burn_tensor =
        Tensor::<B, 1, Float>::from_data(TensorData::new(data, [shape.iter().product()]), device);

    burn_tensor.reshape(Shape::from(shape))
}

/// Convert a NumPy array containing integer data to a Burn tensor

pub fn array_npy2burn_i64<B: Backend, const D: usize>(
    array: &ArrayD<i64>,
    device: &B::Device,
) -> Tensor<B, D, Int> {
    let shape = array.shape().to_vec();

    let data: Vec<i64> = array.iter().cloned().collect();

    let burn_tensor =
        Tensor::<B, 1, Int>::from_data(TensorData::new(data, [shape.iter().product()]), device);

    burn_tensor.reshape(Shape::from(shape))
}
