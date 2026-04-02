use burn::cubecl;
use cubecl::{cube, prelude::*};

const WARP_SIZE: usize = 32;
const VEC: usize = 4;

// CubeCL port of `rapid-sampling` CUDA kernels (`sampling.cu`).
//
// Preserved design points:
// - monotone scan/reduce for threshold statistics;
// - quaternary threshold search in float bit-space;
// - top-k/top-p boundary compensation;
// - two-stage sampling (select tile, then select token within tile).
//
// Adaptations:
// - `Line<f32>` vectorization (CUDA `float4` equivalent);
// - RNG uses LCG (`u32` state) instead of CUDA Philox;
// - keep the `exp()` + `inv_temp` path for now because CubeCL does not expose a direct `exp2`
//   lowering yet; the surrounding launch/indexing/shared-memory structure still follows CUDA.
//
// Performance note:
// The hot path is intentionally a bit repetitive. The original CUDA kernel is tuned around
// 1024 threads/block and a <=64 register/thread budget; extra abstraction that looks cleaner in
// Rust often lengthens live ranges enough to spill on CubeCL CUDA backends.
//
// These kernels also opt into CubeCL fast-math on purpose. The reference CUDA implementation
// relies on relaxed single-precision math as well; forcing stricter libm lowering here costs both
// throughput and register pressure without buying materially better sampling behavior.

#[cube]
fn sanitize_f32(x: f32) -> f32 {
    let zero = f32::new(0.0);
    let max = f32::new(f32::MAX);
    let neg_max = f32::new(-f32::MAX);

    let mut y = if x.is_nan() { zero } else { x };
    if y.is_inf() {
        y = if y < zero { neg_max } else { max };
    }
    y
}

#[cube]
fn lcg_step(z: u32) -> u32 {
    // Same coefficients as cubek-random.
    z * 1664525u32 + 1013904223u32
}

#[cube]
fn u32_to_unit_interval_open(int_random: u32) -> f32 {
    // (0.0, 1.0), matches cubek-random implementation.
    let shifted = int_random >> 9;
    (f32::cast_from(shifted) + f32::new(1.0)) / f32::new(8388609.0) // 2^23 + 1
}

#[cube]
fn warp_reduce_sum_f32(mut val: f32) -> f32 {
    val += plane_shuffle_xor(val, 16);
    val += plane_shuffle_xor(val, 8);
    val += plane_shuffle_xor(val, 4);
    val += plane_shuffle_xor(val, 2);
    val += plane_shuffle_xor(val, 1);
    val
}

#[cube]
fn warp_reduce_sum_u32(mut val: u32) -> u32 {
    val += plane_shuffle_xor(val, 16);
    val += plane_shuffle_xor(val, 8);
    val += plane_shuffle_xor(val, 4);
    val += plane_shuffle_xor(val, 2);
    val += plane_shuffle_xor(val, 1);
    val
}

#[cube]
fn warp_reduce_max_f32(mut val: f32) -> f32 {
    val = max(val, plane_shuffle_xor(val, 16));
    val = max(val, plane_shuffle_xor(val, 8));
    val = max(val, plane_shuffle_xor(val, 4));
    val = max(val, plane_shuffle_xor(val, 2));
    val = max(val, plane_shuffle_xor(val, 1));
    val
}

#[cube]
fn warp_reduce_min_f32(mut val: f32) -> f32 {
    val = min(val, plane_shuffle_xor(val, 16));
    val = min(val, plane_shuffle_xor(val, 8));
    val = min(val, plane_shuffle_xor(val, 4));
    val = min(val, plane_shuffle_xor(val, 2));
    val = min(val, plane_shuffle_xor(val, 1));
    val
}

#[cube]
fn block_reduce_sum_f32(
    mut val: f32,
    mut warp_results: SharedMemory<f32>,
    #[comptime] num_warps: usize,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    val = warp_reduce_sum_f32(val);
    if lane == WARP_SIZE - 1 {
        warp_results[warp_id] = val;
    }
    sync_cube();

    if warp_id == 0 {
        let mut warp_val = if lane < num_warps {
            warp_results[lane]
        } else {
            f32::new(0.0)
        };
        warp_val = warp_reduce_sum_f32(warp_val);
        if lane == 0 {
            warp_results[0] = warp_val;
        }
    }
    sync_cube();
    warp_results[0]
}

#[cube]
fn block_reduce_sum_u32(
    mut val: u32,
    mut warp_results: SharedMemory<u32>,
    #[comptime] num_warps: usize,
) -> u32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    val = warp_reduce_sum_u32(val);
    if lane == WARP_SIZE - 1 {
        warp_results[warp_id] = val;
    }
    sync_cube();

    if warp_id == 0 {
        let mut warp_val = if lane < num_warps {
            warp_results[lane]
        } else {
            0u32.into()
        };
        warp_val = warp_reduce_sum_u32(warp_val);
        if lane == 0 {
            warp_results[0] = warp_val;
        }
    }
    sync_cube();
    warp_results[0]
}

#[cube]
fn block_reduce_sum_f32_monotone(
    mut val: f32,
    mut warp_results: SharedMemory<f32>,
    #[comptime] num_warps: usize,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    val = warp_reduce_sum_f32(val);
    if lane == WARP_SIZE - 1 {
        warp_results[warp_id] = val;
    }
    sync_cube();

    if UNIT_POS == 0 {
        let mut s = warp_results[0];
        #[unroll(true)]
        for i in 1..num_warps {
            s += warp_results[i];
        }
        warp_results[0] = s;
    }
    sync_cube();
    warp_results[0]
}

#[cube]
fn block_reduce_max_f32(
    mut val: f32,
    mut warp_results: SharedMemory<f32>,
    #[comptime] num_warps: usize,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    val = warp_reduce_max_f32(val);
    if lane == WARP_SIZE - 1 {
        warp_results[warp_id] = val;
    }
    sync_cube();

    if warp_id == 0 {
        let mut warp_val = if lane < num_warps {
            warp_results[lane]
        } else {
            f32::new(-f32::MAX)
        };
        warp_val = warp_reduce_max_f32(warp_val);
        if lane == 0 {
            warp_results[0] = warp_val;
        }
    }
    sync_cube();
    warp_results[0]
}

#[cube]
fn block_reduce_min_f32(
    mut val: f32,
    mut warp_results: SharedMemory<f32>,
    #[comptime] num_warps: usize,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    val = warp_reduce_min_f32(val);
    if lane == WARP_SIZE - 1 {
        warp_results[warp_id] = val;
    }
    sync_cube();

    if warp_id == 0 {
        let mut warp_val = if lane < num_warps {
            warp_results[lane]
        } else {
            f32::new(f32::MAX)
        };
        warp_val = warp_reduce_min_f32(warp_val);
        if lane == 0 {
            warp_results[0] = warp_val;
        }
    }
    sync_cube();
    warp_results[0]
}

#[cube]
fn warp_inclusive_scan_f32(mut val: f32) -> f32 {
    let lane = UNIT_POS_PLANE;

    let n = plane_shuffle_up(val, 16);
    if lane >= 16 {
        val += n;
    }
    let n = plane_shuffle_up(val, 8);
    if lane >= 8 {
        val += n;
    }
    let n = plane_shuffle_up(val, 4);
    if lane >= 4 {
        val += n;
    }
    let n = plane_shuffle_up(val, 2);
    if lane >= 2 {
        val += n;
    }
    let n = plane_shuffle_up(val, 1);
    if lane >= 1 {
        val += n;
    }

    val
}

#[cube]
fn warp_inclusive_scan_u32(mut val: u32) -> u32 {
    let lane = UNIT_POS_PLANE;

    let n = plane_shuffle_up(val, 16);
    if lane >= 16 {
        val += n;
    }
    let n = plane_shuffle_up(val, 8);
    if lane >= 8 {
        val += n;
    }
    let n = plane_shuffle_up(val, 4);
    if lane >= 4 {
        val += n;
    }
    let n = plane_shuffle_up(val, 2);
    if lane >= 2 {
        val += n;
    }
    let n = plane_shuffle_up(val, 1);
    if lane >= 1 {
        val += n;
    }

    val
}

#[cube]
fn block_inclusive_scan_f32(
    val: f32,
    mut warp_sums: SharedMemory<f32>,
    mut total_out: Shared<f32>,
    #[comptime] num_warps: usize,
    #[comptime] block_size: usize,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    let mut val1 = warp_inclusive_scan_f32(val);

    if lane == WARP_SIZE - 1 {
        warp_sums[warp_id] = val1;
    }
    sync_cube();

    // MUST sum this way to ensure numerical monotonicity, not merely "better stability".
    // A more symmetric warp-scan of warp totals looks nicer but breaks the monotone cumulative
    // property relied on by the original thresholding and selection logic.
    if UNIT_POS == 0 {
        let mut s = warp_sums[0];
        #[unroll(true)]
        for i in 1..num_warps {
            s += warp_sums[i];
            warp_sums[i] = s;
        }
    }
    sync_cube();

    if warp_id > 0 {
        val1 += warp_sums[warp_id - 1];
    }

    if UNIT_POS as usize == block_size - 1 {
        *total_out.as_mut() = val1;
    }
    sync_cube();

    val1
}

#[cube]
fn block_inclusive_scan_u32(
    val: u32,
    mut warp_sums: SharedMemory<u32>,
    mut total_out: Shared<u32>,
    #[comptime] num_warps: usize,
    #[comptime] block_size: usize,
) -> u32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    let mut val1 = warp_inclusive_scan_u32(val);

    if lane == WARP_SIZE - 1 {
        warp_sums[warp_id] = val1;
    }
    sync_cube();

    if UNIT_POS == 0 {
        let mut s = warp_sums[0];
        #[unroll(true)]
        for i in 1..num_warps {
            s += warp_sums[i];
            warp_sums[i] = s;
        }
    }
    sync_cube();

    if warp_id > 0 {
        val1 += warp_sums[warp_id - 1];
    }

    if UNIT_POS as usize == block_size - 1 {
        *total_out.as_mut() = val1;
    }
    sync_cube();

    val1
}

#[cube]
fn select_boundary_thread(
    u: bool,
    mut warp_last_lane: SharedMemory<u32>,
    #[comptime] _num_warps: usize,
) -> bool {
    // Returns true iff this thread is the first where `u` becomes true.
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    let u32_val: u32 = if u { 1u32.into() } else { 0u32.into() };

    if lane == WARP_SIZE - 1 {
        warp_last_lane[warp_id] = u32_val;
    }
    sync_cube();

    let mut prev_u = plane_shuffle_up(u32_val, 1);
    if lane == 0 {
        prev_u = if warp_id == 0 {
            0u32.into()
        } else {
            warp_last_lane[warp_id - 1]
        };
    }
    sync_cube();

    u32_val != prev_u
}

#[derive(CubeLaunch, CubeType)]
pub struct RapidSampleBatchIdsInputs {
    pub logits: Tensor<Line<f32>>,
    pub batch_ids: Tensor<u32>,
}

#[derive(CubeLaunch, CubeType)]
pub struct RapidSampleRepetitionOutputs {
    pub token_ids: Tensor<i32>,
    pub penalties: Tensor<Line<f32>>,
    pub states: Tensor<u32>,
    pub probs: Tensor<Line<f32>>,
}

#[derive(CubeLaunch, CubeType)]
pub struct RapidSamplePenaltyParams {
    pub inv_temp: Tensor<f32>,           // [batch_size]
    pub top_k: Tensor<u32>,              // [batch_size]
    pub top_p: Tensor<f32>,              // [batch_size]
    pub presence_penalty: Tensor<f32>,   // [batch_size]
    pub repetition_penalty: Tensor<f32>, // [batch_size]
    pub penalty_decay: Tensor<f32>,      // [batch_size]
}

#[derive(CubeLaunch, CubeType, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RapidSampleConfig {
    pub block_size: usize,
    pub num_warps: usize,
}

#[derive(CubeType, Clone, Copy)]
struct ThresholdStats {
    prob_sum_gt_threshold: f32,
    count_eq_threshold: u32,
    count_gt_threshold: u32,
}

#[cube]
fn search_threshold_prob(
    probs: &Tensor<Line<f32>>,
    batch_base_vec: usize,
    vocab_size_vec: u32,
    unit_index: u32,
    block_size: u32,
    vocab_size: u32,
    top_k: u32,
    top_p: f32,
    prob_min: f32,
    prob_max: f32,
    warp_f32: SharedMemory<f32>,
    warp_u32: SharedMemory<u32>,
    #[comptime] num_warps: usize,
) -> f32 {
    let mut threshold_bits_left = u32::reinterpret(prob_min);
    let mut threshold_bits_right = u32::reinterpret(prob_max) + 1;

    let mut left_candidate_count = vocab_size;
    let mut left_candidate_prob_sum = f32::new(1.0);

    while (left_candidate_count > top_k || left_candidate_prob_sum > top_p)
        && threshold_bits_left < threshold_bits_right - 1
    {
        // This is intentionally quaternary, not binary. Paying three comparisons per read round
        // is cheaper than another full global-memory sweep.
        let pivot_left = threshold_bits_left;
        let pivot_mid = (threshold_bits_left + threshold_bits_right) / 2;
        let pivot_lower_mid = (threshold_bits_left + pivot_mid) / 2;
        let pivot_upper_mid = (pivot_mid + threshold_bits_right) / 2;

        let pivot_lower_mid_prob = f32::reinterpret(pivot_lower_mid);
        let pivot_mid_prob = f32::reinterpret(pivot_mid);
        let pivot_upper_mid_prob = f32::reinterpret(pivot_upper_mid);

        let mut local_count_lower_mid = 0u32;
        let mut local_count_mid = 0u32;
        let mut local_count_upper_mid = 0u32;

        let mut local_sum_lower_mid = f32::new(0.0);
        let mut local_sum_mid = f32::new(0.0);
        let mut local_sum_upper_mid = f32::new(0.0);

        let mut vec_index = unit_index;
        while vec_index < vocab_size_vec {
            let prob_vec = probs[batch_base_vec + vec_index as usize];
            #[unroll(true)]
            for lane in 0..VEC {
                let prob = prob_vec[lane];
                if prob >= pivot_lower_mid_prob {
                    local_count_lower_mid += 1;
                    local_sum_lower_mid += prob;
                }
                if prob >= pivot_mid_prob {
                    local_count_mid += 1;
                    local_sum_mid += prob;
                }
                if prob >= pivot_upper_mid_prob {
                    local_count_upper_mid += 1;
                    local_sum_upper_mid += prob;
                }
            }
            vec_index += block_size;
        }

        let global_count_lower_mid =
            block_reduce_sum_u32(local_count_lower_mid, warp_u32, num_warps);
        let global_sum_lower_mid =
            block_reduce_sum_f32_monotone(local_sum_lower_mid, warp_f32, num_warps);

        if global_count_lower_mid < top_k && global_sum_lower_mid < top_p {
            threshold_bits_left = pivot_left;
            threshold_bits_right = pivot_lower_mid;
        } else {
            let global_count_mid = block_reduce_sum_u32(local_count_mid, warp_u32, num_warps);
            let global_sum_mid = block_reduce_sum_f32_monotone(local_sum_mid, warp_f32, num_warps);

            if global_count_mid < top_k && global_sum_mid < top_p {
                threshold_bits_left = pivot_lower_mid;
                threshold_bits_right = pivot_mid;
                left_candidate_count = global_count_lower_mid;
                left_candidate_prob_sum = global_sum_lower_mid;
            } else {
                let global_count_upper_mid =
                    block_reduce_sum_u32(local_count_upper_mid, warp_u32, num_warps);
                let global_sum_upper_mid =
                    block_reduce_sum_f32_monotone(local_sum_upper_mid, warp_f32, num_warps);

                if global_count_upper_mid < top_k && global_sum_upper_mid < top_p {
                    threshold_bits_left = pivot_mid;
                    threshold_bits_right = pivot_upper_mid;
                    left_candidate_count = global_count_mid;
                    left_candidate_prob_sum = global_sum_mid;
                } else {
                    threshold_bits_left = pivot_upper_mid;
                    left_candidate_count = global_count_upper_mid;
                    left_candidate_prob_sum = global_sum_upper_mid;
                }
            }
        }
    }

    f32::reinterpret(threshold_bits_left)
}

#[cube]
fn compute_threshold_stats(
    probs: &Tensor<Line<f32>>,
    batch_base_vec: usize,
    vocab_size_vec: u32,
    unit_index: u32,
    block_size: u32,
    threshold_prob: f32,
) -> ThresholdStats {
    let mut local_prob_sum_gt_threshold = f32::new(0.0);
    let mut local_count_eq_threshold = 0u32;
    let mut local_count_gt_threshold = 0u32;

    let mut vec_index = unit_index;
    while vec_index < vocab_size_vec {
        let prob_vec = probs[batch_base_vec + vec_index as usize];
        #[unroll(true)]
        for lane in 0..VEC {
            let prob = prob_vec[lane];
            if prob == threshold_prob {
                local_count_eq_threshold += 1;
            }
            if prob > threshold_prob {
                local_count_gt_threshold += 1;
                local_prob_sum_gt_threshold += prob;
            }
        }
        vec_index += block_size;
    }

    ThresholdStats {
        prob_sum_gt_threshold: local_prob_sum_gt_threshold,
        count_eq_threshold: local_count_eq_threshold,
        count_gt_threshold: local_count_gt_threshold,
    }
}

#[cube]
fn compute_threshold_compensation(
    threshold_stats: ThresholdStats,
    threshold_prob: f32,
    top_k: u32,
    top_p: f32,
    warp_f32: SharedMemory<f32>,
    warp_u32: SharedMemory<u32>,
    #[comptime] num_warps: usize,
    #[comptime] block_size: usize,
) -> f32 {
    let scan_f32 = warp_f32;
    let scan_u32 = warp_u32;
    // These are true shared scalars on purpose. `SharedMemory::new(1)` is functionally equivalent
    // but maps to a tiny shared array, which is not what the CUDA kernel uses and tends to bloat
    // allocator state / register pressure in CubeCL codegen.
    let shared_prob_sum_gt_threshold = Shared::<f32>::new();
    let shared_count_eq_threshold = Shared::<u32>::new();
    let shared_count_gt_threshold = Shared::<u32>::new();

    let _ = block_inclusive_scan_f32(
        threshold_stats.prob_sum_gt_threshold,
        scan_f32,
        shared_prob_sum_gt_threshold,
        num_warps,
        block_size,
    );
    let _ = block_inclusive_scan_u32(
        threshold_stats.count_eq_threshold,
        scan_u32,
        shared_count_eq_threshold,
        num_warps,
        block_size,
    );
    let _ = block_inclusive_scan_u32(
        threshold_stats.count_gt_threshold,
        scan_u32,
        shared_count_gt_threshold,
        num_warps,
        block_size,
    );

    let total_prob_sum_gt_threshold = *shared_prob_sum_gt_threshold.as_ref();
    let total_count_eq_threshold = *shared_count_eq_threshold.as_ref();
    let total_count_gt_threshold = *shared_count_gt_threshold.as_ref();
    let mut eq_threshold_compensation = f32::new(1.0);
    if total_count_eq_threshold > 0 {
        let threshold_mass = threshold_prob * f32::cast_from(total_count_eq_threshold);
        if threshold_mass != f32::new(0.0) {
            eq_threshold_compensation = min(
                eq_threshold_compensation,
                sanitize_f32((top_p - total_prob_sum_gt_threshold) / threshold_mass),
            );
        }
        eq_threshold_compensation = min(
            eq_threshold_compensation,
            sanitize_f32(
                (f32::cast_from(top_k) - f32::cast_from(total_count_gt_threshold))
                    / f32::cast_from(total_count_eq_threshold),
            ),
        );
        eq_threshold_compensation = max(eq_threshold_compensation, f32::new(0.0));
    }

    eq_threshold_compensation
}

#[cube]
fn sample_from_threshold(
    probs: &Tensor<Line<f32>>,
    token_ids: &mut Tensor<i32>,
    states: &mut Tensor<u32>,
    batch_index: usize,
    state_index: usize,
    batch_base_vec: usize,
    vocab_size: u32,
    unit_index: u32,
    threshold_stats: ThresholdStats,
    threshold_prob: f32,
    eq_threshold_compensation: f32,
    warp_f32: SharedMemory<f32>,
    warp_u32: SharedMemory<u32>,
    #[comptime] num_warps: usize,
    #[comptime] block_size: usize,
) -> u32 {
    let scan_f32 = warp_f32;

    let shared_total_selected_mass = Shared::<f32>::new();
    let local_selected_mass = threshold_stats.prob_sum_gt_threshold
        + (threshold_prob * f32::cast_from(threshold_stats.count_eq_threshold))
            * eq_threshold_compensation;
    let cumulative_selected_mass = block_inclusive_scan_f32(
        local_selected_mass,
        scan_f32,
        shared_total_selected_mass,
        num_warps,
        block_size,
    );

    let mut shared_random_v = Shared::<f32>::new();
    let mut shared_random_target_prob = Shared::<f32>::new();
    let mut shared_selected_tile_unit_index = Shared::<u32>::new();

    if UNIT_POS == 0 {
        *shared_selected_tile_unit_index.as_mut() = 0u32;

        let mut rng_state = states[state_index];
        rng_state = lcg_step(rng_state);
        let random_u = u32_to_unit_interval_open(rng_state);
        rng_state = lcg_step(rng_state);
        let random_v = u32_to_unit_interval_open(rng_state);

        states[state_index] = rng_state;
        *shared_random_v.as_mut() = random_v;
        *shared_random_target_prob.as_mut() = *shared_total_selected_mass.as_ref() * random_u;
    }
    sync_cube();

    let should_select_tile = *shared_random_target_prob.as_ref() <= cumulative_selected_mass;
    let is_tile_boundary = select_boundary_thread(should_select_tile, warp_u32, num_warps);
    if is_tile_boundary {
        *shared_selected_tile_unit_index.as_mut() = UNIT_POS;
    }
    sync_cube();

    let selected_tile_unit_index = *shared_selected_tile_unit_index.as_ref();
    // This mapping is intentionally odd. It is the second stage of the original 1024-thread
    // sampler: first pick the responsible thread, then let every lane cover one token inside that
    // thread's stripe. "Simplifying" it to linear tile indexing breaks the block_size^2 coverage.
    let candidate_token_id = selected_tile_unit_index * VEC as u32
        + (unit_index / VEC as u32) * (VEC as u32) * (block_size as u32)
        + (unit_index % VEC as u32);

    let candidate_prob = if candidate_token_id < vocab_size {
        let candidate_token_id_usize = candidate_token_id as usize;
        let candidate_prob_vec = probs[batch_base_vec + (candidate_token_id_usize / VEC)];
        candidate_prob_vec[candidate_token_id_usize % VEC]
    } else {
        f32::new(0.0)
    };

    let adjusted_candidate_prob = if candidate_prob < threshold_prob {
        f32::new(0.0)
    } else if candidate_prob == threshold_prob {
        candidate_prob * eq_threshold_compensation
    } else {
        candidate_prob
    };

    let shared_total_adjusted_prob = Shared::<f32>::new();
    let cumulative_adjusted_prob = block_inclusive_scan_f32(
        adjusted_candidate_prob,
        scan_f32,
        shared_total_adjusted_prob,
        num_warps,
        block_size,
    );

    let random_second = *shared_total_adjusted_prob.as_ref() * *shared_random_v.as_ref();
    let should_select_token = random_second <= cumulative_adjusted_prob;

    let mut shared_sampled_token_id = Shared::<u32>::new();
    if UNIT_POS == 0 {
        *shared_sampled_token_id.as_mut() = 0u32;
    }
    sync_cube();

    let is_token_boundary = select_boundary_thread(should_select_token, warp_u32, num_warps);
    if is_token_boundary {
        *shared_sampled_token_id.as_mut() = if candidate_token_id < vocab_size {
            candidate_token_id
        } else {
            0u32.into()
        };
    }
    sync_cube();

    if UNIT_POS == 0 {
        token_ids[batch_index] = i32::cast_from(*shared_sampled_token_id.as_ref());
    }

    *shared_sampled_token_id.as_ref()
}

#[cube(launch, fast_math = FastMath::all())]
pub fn rapid_sample_repetition_temperature_topk_topp_batch_ids_kernel(
    inputs: &RapidSampleBatchIdsInputs,
    outputs: &mut RapidSampleRepetitionOutputs,
    params: &RapidSamplePenaltyParams,
    vocab_size: u32,
    #[comptime] config: RapidSampleConfig,
) {
    let block_size = comptime![config.block_size];
    let num_warps = comptime![config.num_warps];
    let block_size_u32 = block_size as u32;

    let batch_index = CUBE_POS_X as usize;
    let unit_index = UNIT_POS;

    if unit_index >= block_size_u32 {
        terminate!();
    }

    let state_index = inputs.batch_ids[batch_index] as usize;
    let vocab_size_vec = vocab_size / VEC as u32;
    let probs_batch_base_vec = batch_index * vocab_size_vec as usize;
    let penalties_batch_base_vec = state_index * vocab_size_vec as usize;

    let shared_warp_f32 = SharedMemory::<f32>::new(num_warps);
    let shared_warp_u32 = SharedMemory::<u32>::new(num_warps);

    // This mapping-oriented kernel exists for end-to-end decode throughput, not aesthetics.
    // The queue state lives in full-size `[max_batch, vocab]` buffers, so gathering and scattering
    // penalties around every token step duplicates the sampler's entire memory traffic. Passing the
    // active-row -> state-slot mapping into the kernel lets us update those full buffers in place.

    let global_scaled_logit_max = {
        let inv_temp = params.inv_temp[batch_index];
        let mut local_scaled_logit_max = f32::new(-f32::MAX);
        let mut vec_index = unit_index;
        while vec_index < vocab_size_vec {
            let probs_tensor_index = probs_batch_base_vec + vec_index as usize;
            let penalties_tensor_index = penalties_batch_base_vec + vec_index as usize;
            let mut logit_vec = inputs.logits[probs_tensor_index];
            let penalty_vec = outputs.penalties[penalties_tensor_index];

            #[unroll(true)]
            for lane in 0..VEC {
                let scaled_logit =
                    sanitize_f32((sanitize_f32(logit_vec[lane]) - penalty_vec[lane]) * inv_temp);
                logit_vec[lane] = scaled_logit;
                local_scaled_logit_max = max(local_scaled_logit_max, scaled_logit);
            }

            outputs.probs[probs_tensor_index] = logit_vec;
            vec_index += block_size_u32;
        }

        block_reduce_max_f32(local_scaled_logit_max, shared_warp_f32, num_warps)
    };

    let exp_sum_denom = {
        let mut local_exp_sum = f32::new(0.0);
        let mut vec_index = unit_index;
        while vec_index < vocab_size_vec {
            let scaled_logit_vec = outputs.probs[probs_batch_base_vec + vec_index as usize];
            #[unroll(true)]
            for lane in 0..VEC {
                local_exp_sum += (scaled_logit_vec[lane] - global_scaled_logit_max).exp();
            }
            vec_index += block_size_u32;
        }
        block_reduce_sum_f32(local_exp_sum, shared_warp_f32, num_warps)
    };

    let (prob_max, prob_min) = {
        let mut local_prob_max = f32::new(-f32::MAX);
        let mut local_prob_min = f32::new(f32::MAX);
        let mut vec_index = unit_index;
        while vec_index < vocab_size_vec {
            let probs_tensor_index = probs_batch_base_vec + vec_index as usize;
            let mut prob_vec = outputs.probs[probs_tensor_index];
            #[unroll(true)]
            for lane in 0..VEC {
                let prob = (prob_vec[lane] - global_scaled_logit_max).exp() / exp_sum_denom;
                prob_vec[lane] = prob;
                local_prob_max = max(local_prob_max, prob);
                local_prob_min = min(local_prob_min, prob);
            }
            outputs.probs[probs_tensor_index] = prob_vec;
            vec_index += block_size_u32;
        }
        (
            block_reduce_max_f32(local_prob_max, shared_warp_f32, num_warps),
            block_reduce_min_f32(local_prob_min, shared_warp_f32, num_warps),
        )
    };

    let top_k = params.top_k[batch_index];
    let top_p = params.top_p[batch_index];
    let threshold_prob = search_threshold_prob(
        &outputs.probs,
        probs_batch_base_vec,
        vocab_size_vec,
        unit_index,
        block_size_u32,
        vocab_size,
        top_k,
        top_p,
        prob_min,
        prob_max,
        shared_warp_f32,
        shared_warp_u32,
        num_warps,
    );

    let threshold_stats = compute_threshold_stats(
        &outputs.probs,
        probs_batch_base_vec,
        vocab_size_vec,
        unit_index,
        block_size_u32,
        threshold_prob,
    );
    let eq_threshold_compensation = compute_threshold_compensation(
        threshold_stats,
        threshold_prob,
        top_k,
        top_p,
        shared_warp_f32,
        shared_warp_u32,
        num_warps,
        block_size,
    );

    let sampled_token_id = sample_from_threshold(
        &outputs.probs,
        &mut outputs.token_ids,
        &mut outputs.states,
        batch_index,
        state_index,
        probs_batch_base_vec,
        vocab_size,
        unit_index,
        threshold_stats,
        threshold_prob,
        eq_threshold_compensation,
        shared_warp_f32,
        shared_warp_u32,
        num_warps,
        block_size,
    );

    let presence_penalty = params.presence_penalty[batch_index];
    let repetition_penalty = params.repetition_penalty[batch_index];
    let penalty_decay = params.penalty_decay[batch_index];
    let mut vec_index = unit_index;
    while vec_index < vocab_size_vec {
        let penalties_tensor_index = penalties_batch_base_vec + vec_index as usize;
        let mut penalty_vec = outputs.penalties[penalties_tensor_index];
        #[unroll(true)]
        for lane in 0..VEC {
            let token_id = vec_index * VEC as u32 + lane as u32;
            let current_penalty = penalty_vec[lane];
            let penalty_add = if sampled_token_id == token_id {
                if current_penalty == f32::new(0.0) {
                    presence_penalty
                } else {
                    repetition_penalty
                }
            } else {
                f32::new(0.0)
            };
            penalty_vec[lane] = fma(current_penalty, penalty_decay, penalty_add);
        }
        outputs.penalties[penalties_tensor_index] = penalty_vec;
        vec_index += block_size_u32;
    }
}
