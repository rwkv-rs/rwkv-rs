use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::CubeRuntime;

use crate::kernels::rapid_sample::{
    RapidSampleBackend, RapidSampleOutputPrimitive, RapidSamplePenaltyConfig,
    host::rapid_sample_topk_topp_impl,
};

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> RapidSampleBackend
    for CubeBackend<R, F, I, BT>
{
    fn rapid_sample(
        logits: FloatTensor<Self>,
        states: IntTensor<Self>,
        temperature: f32,
        top_k: i32,
        top_p: f32,
        penalties: Option<(FloatTensor<Self>, RapidSamplePenaltyConfig)>,
    ) -> RapidSampleOutputPrimitive<Self> {
        rapid_sample_topk_topp_impl::<R, F, I, BT>(
            logits,
            states,
            temperature,
            top_k,
            top_p,
            penalties,
        )
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::{DType, Shape};
    use burn_fusion::{
        stream::{Operation, OperationStreams}, Fusion, FusionBackend,
        FusionRuntime,
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;
    use crate::kernels::rapid_sample::{
        RapidSampleBackend, RapidSampleOutput, RapidSampleOutputPrimitive, RapidSamplePenaltyConfig,
    };

    impl<B: FusionBackend + RapidSampleBackend> RapidSampleBackend for Fusion<B> {
        fn rapid_sample(
            logits: FloatTensor<Self>,
            states: IntTensor<Self>,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            penalties: Option<(FloatTensor<Self>, RapidSamplePenaltyConfig)>,
        ) -> RapidSampleOutputPrimitive<Self> {
            let client = logits.client.clone();
            let batch_size = logits.shape[0];
            let vocab_size = logits.shape[1];

            match penalties {
                None => {
                    #[derive(Clone, Debug)]
                    struct RapidSampleOp<B1> {
                        desc: CustomOpIr,
                        temperature: f32,
                        top_k: i32,
                        top_p: f32,
                        _b: core::marker::PhantomData<B1>,
                    }

                    impl<B1: FusionBackend + RapidSampleBackend> Operation<B1::FusionRuntime>
                    for RapidSampleOp<B1> {
                        fn execute(
                            &self,
                            handles: &mut HandleContainer<
                                <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                            >,
                        ) {
                            let ([logits, states], [token_ids_out, states_out]) = self.desc.as_fixed();

                            let logits_tensor = handles.get_float_tensor::<B1>(logits);
                            let states_tensor = handles.get_int_tensor::<B1>(states);

                            let output = B1::rapid_sample(
                                logits_tensor,
                                states_tensor,
                                self.temperature,
                                self.top_k,
                                self.top_p,
                                None,
                            );

                            handles.register_int_tensor::<B1>(&token_ids_out.id, output.token_ids);
                            handles.register_int_tensor::<B1>(&states_out.id, output.states);
                        }
                    }

                    let mut streams = OperationStreams::default();
                    streams.tensor(&logits);
                    streams.tensor(&states);

                    let output_desc = [
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size]),
                            DType::I32,
                        ),
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size]),
                            DType::U32,
                        ),
                    ];

                    let desc = CustomOpIr::new(
                        "rapid_sample",
                        &[logits.into_ir(), states.into_ir()],
                        &output_desc,
                    );

                    let op = RapidSampleOp::<B> {
                        desc,
                        temperature,
                        top_k,
                        top_p,
                        _b: core::marker::PhantomData,
                    };

                    let mut outputs =
                        client.register(streams, OperationIr::Custom(op.desc.clone()), op);

                    let states = outputs.pop().expect("missing states");
                    let token_ids = outputs.pop().expect("missing token_ids");

                    RapidSampleOutput {
                        token_ids,
                        states,
                        penalties: None,
                    }
                }
                Some((penalties, penalty_cfg)) => {
                    #[derive(Clone, Debug)]
                    struct RapidSamplePenaltyOp<B1> {
                        desc: CustomOpIr,
                        temperature: f32,
                        top_k: i32,
                        top_p: f32,
                        penalty_cfg: RapidSamplePenaltyConfig,
                        _b: core::marker::PhantomData<B1>,
                    }

                    impl<B1: FusionBackend + RapidSampleBackend> Operation<B1::FusionRuntime>
                    for RapidSamplePenaltyOp<B1> {
                        fn execute(
                            &self,
                            handles: &mut HandleContainer<
                                <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                            >,
                        ) {
                            let ([logits, states, penalties], [token_ids_out, states_out, penalties_out]) =
                                self.desc.as_fixed();

                            let logits_tensor = handles.get_float_tensor::<B1>(logits);
                            let states_tensor = handles.get_int_tensor::<B1>(states);
                            let penalties_tensor = handles.get_float_tensor::<B1>(penalties);

                            let output = B1::rapid_sample(
                                logits_tensor,
                                states_tensor,
                                self.temperature,
                                self.top_k,
                                self.top_p,
                                Some((penalties_tensor, self.penalty_cfg)),
                            );

                            handles.register_int_tensor::<B1>(&token_ids_out.id, output.token_ids);
                            handles.register_int_tensor::<B1>(&states_out.id, output.states);
                            handles.register_float_tensor::<B1>(
                                &penalties_out.id,
                                output.penalties.expect("penalties output required"),
                            );
                        }
                    }

                    let mut streams = OperationStreams::default();
                    streams.tensor(&logits);
                    streams.tensor(&states);
                    streams.tensor(&penalties);

                    let output_desc = [
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size]),
                            DType::I32,
                        ),
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size]),
                            DType::U32,
                        ),
                        TensorIr::uninit(
                            client.create_empty_handle(),
                            Shape::new([batch_size, vocab_size]),
                            DType::F32,
                        ),
                    ];

                    let desc = CustomOpIr::new(
                        "rapid_sample_penalty",
                        &[logits.into_ir(), states.into_ir(), penalties.into_ir()],
                        &output_desc,
                    );

                    let op = RapidSamplePenaltyOp::<B> {
                        desc,
                        temperature,
                        top_k,
                        top_p,
                        penalty_cfg,
                        _b: core::marker::PhantomData,
                    };

                    let mut outputs =
                        client.register(streams, OperationIr::Custom(op.desc.clone()), op);

                    let penalties = outputs.pop().expect("missing penalties");
                    let states = outputs.pop().expect("missing states");
                    let token_ids = outputs.pop().expect("missing token_ids");

                    RapidSampleOutput {
                        token_ids,
                        states,
                        penalties: Some(penalties),
                    }
                }
            }
        }
    }
}
