use burn::backend::wgpu::{BoolElement, CubeBackend, FloatElement, IntElement};
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::CubeRuntime;

use crate::kernels::rapid_sample::{
    RapidSampleBackend, RapidSampleOutputPrimitive,
    host::rapid_sample_topk_topp_impl,
};

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> RapidSampleBackend
    for CubeBackend<R, F, I, BT>
{
    fn rapid_sample(
        logits: FloatTensor<Self>,
        states: IntTensor<Self>,
        inv_temperatures: FloatTensor<Self>,
        top_ks: IntTensor<Self>,
        top_ps: FloatTensor<Self>,
        penalties: Option<(FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>)>,
    ) -> RapidSampleOutputPrimitive<Self> {
        rapid_sample_topk_topp_impl::<R, F, I, BT>(
            logits,
            states,
            inv_temperatures,
            top_ks,
            top_ps,
            penalties,
        )
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::{DType, Shape};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;
    use crate::kernels::rapid_sample::{
        RapidSampleBackend, RapidSampleOutput, RapidSampleOutputPrimitive,
    };

    impl<B: FusionBackend + RapidSampleBackend> RapidSampleBackend for Fusion<B> {
        fn rapid_sample(
            logits: FloatTensor<Self>,
            states: IntTensor<Self>,
            inv_temperatures: FloatTensor<Self>,
            top_ks: IntTensor<Self>,
            top_ps: FloatTensor<Self>,
            penalties: Option<(FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>)>,
        ) -> RapidSampleOutputPrimitive<Self> {
            let client = logits.client.clone();
            let batch_size = logits.shape[0];
            let vocab_size = logits.shape[1];

            match penalties {
                None => {
                    #[derive(Clone, Debug)]
                    struct RapidSampleOp<B1> {
                        desc: CustomOpIr,
                        _b: core::marker::PhantomData<B1>,
                    }

                    impl<B1: FusionBackend + RapidSampleBackend> Operation<B1::FusionRuntime> for RapidSampleOp<B1> {
                        fn execute(
                            &self,
                            handles: &mut HandleContainer<
                                <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                            >,
                        ) {
                            let (
                                [logits, states, inv_temperatures, top_ks, top_ps],
                                [token_ids_out, states_out],
                            ) = self.desc.as_fixed();

                            let logits_tensor = handles.get_float_tensor::<B1>(logits);
                            let states_tensor = handles.get_int_tensor::<B1>(states);
                            let inv_temp_tensor = handles.get_float_tensor::<B1>(inv_temperatures);
                            let top_ks_tensor = handles.get_int_tensor::<B1>(top_ks);
                            let top_ps_tensor = handles.get_float_tensor::<B1>(top_ps);

                            let output = B1::rapid_sample(
                                logits_tensor,
                                states_tensor,
                                inv_temp_tensor,
                                top_ks_tensor,
                                top_ps_tensor,
                                None,
                            );

                            handles.register_int_tensor::<B1>(&token_ids_out.id, output.token_ids);
                            handles.register_int_tensor::<B1>(&states_out.id, output.states);
                        }
                    }

                    let mut streams = OperationStreams::default();
                    streams.tensor(&logits);
                    streams.tensor(&states);
                    streams.tensor(&inv_temperatures);
                    streams.tensor(&top_ks);
                    streams.tensor(&top_ps);

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
                        &[
                            logits.into_ir(),
                            states.into_ir(),
                            inv_temperatures.into_ir(),
                            top_ks.into_ir(),
                            top_ps.into_ir(),
                        ],
                        &output_desc,
                    );

                    let op = RapidSampleOp::<B> {
                        desc,
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
                Some((penalties, presence_penalty, repetition_penalty, penalty_decay)) => {
                    #[derive(Clone, Debug)]
                    struct RapidSamplePenaltyOp<B1> {
                        desc: CustomOpIr,
                        _b: core::marker::PhantomData<B1>,
                    }

                    impl<B1: FusionBackend + RapidSampleBackend> Operation<B1::FusionRuntime>
                        for RapidSamplePenaltyOp<B1>
                    {
                        fn execute(
                            &self,
                            handles: &mut HandleContainer<
                                <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                            >,
                        ) {
                            let (
                                [logits, states, inv_temperatures, top_ks, top_ps, penalties, pp, rp, pd],
                                [token_ids_out, states_out, penalties_out],
                            ) = self.desc.as_fixed();

                            let logits_tensor = handles.get_float_tensor::<B1>(logits);
                            let states_tensor = handles.get_int_tensor::<B1>(states);
                            let inv_temp_tensor = handles.get_float_tensor::<B1>(inv_temperatures);
                            let top_ks_tensor = handles.get_int_tensor::<B1>(top_ks);
                            let top_ps_tensor = handles.get_float_tensor::<B1>(top_ps);
                            let penalties_tensor = handles.get_float_tensor::<B1>(penalties);
                            let pp_tensor = handles.get_float_tensor::<B1>(pp);
                            let rp_tensor = handles.get_float_tensor::<B1>(rp);
                            let pd_tensor = handles.get_float_tensor::<B1>(pd);

                            let output = B1::rapid_sample(
                                logits_tensor,
                                states_tensor,
                                inv_temp_tensor,
                                top_ks_tensor,
                                top_ps_tensor,
                                Some((penalties_tensor, pp_tensor, rp_tensor, pd_tensor)),
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
                    streams.tensor(&inv_temperatures);
                    streams.tensor(&top_ks);
                    streams.tensor(&top_ps);
                    streams.tensor(&penalties);
                    streams.tensor(&presence_penalty);
                    streams.tensor(&repetition_penalty);
                    streams.tensor(&penalty_decay);

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
                        &[
                            logits.into_ir(),
                            states.into_ir(),
                            inv_temperatures.into_ir(),
                            top_ks.into_ir(),
                            top_ps.into_ir(),
                            penalties.into_ir(),
                            presence_penalty.into_ir(),
                            repetition_penalty.into_ir(),
                            penalty_decay.into_ir(),
                        ],
                        &output_desc,
                    );

                    let op = RapidSamplePenaltyOp::<B> {
                        desc,
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
