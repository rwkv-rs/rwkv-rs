// use std::time::Duration;
// use axum::extract::State;
// use axum::http::HeaderMap;
// use axum::Json;
// use axum::response::{IntoResponse, Response, Sse};
// use axum::response::sse::{Event, KeepAlive};
// use tokio::sync::mpsc;
// use tokio_stream::wrappers::ReceiverStream;
// use crate::dtos::chat::completions::ChatCompletionsReq;
// use crate::routes::AppState;
//
// #[cfg_attr(
//     feature = "trace",
//     tracing::instrument(
//         name = "rwkv.infer.http.chat_completions",
//         skip_all,
//         fields(path = "/v1/chat/completions")
//     )
// )]
// pub async fn chat_completions(
//     headers: HeaderMap,
//     State(app_state): State<AppState>,
//     Json(req): Json<ChatCompletionsReq>,
// ) -> Response {
//     if let Err(resp) = check_api_key(&headers, &app_state.auth_cfg) {
//         #[cfg(feature = "trace")]
//         tracing::warn!("api key check failed");
//         return resp;
//     }
//
//     let api = HttpApiService::new(app_state.runtime_manager.clone());
//     let run = match api.chat_completions(req).await {
//         Ok(run) => run,
//         Err(e) => return crate::access::http_api::infer_error_response(e),
//     };
//
//     if run.stream_requested {
//         let (sse_tx, sse_rx) = mpsc::channel(256);
//         tokio::spawn(async move {
//             let mut run = run;
//             #[cfg(feature = "trace")]
//             let mut emitted_chunks = 0usize;
//             #[cfg(feature = "trace")]
//             tracing::info!(chat_completion_id = %run.id, model = %run.model, "chat stream opened");
//
//             let role_chunk = sonic_rs::to_string(&run.stream_role_chunk()).unwrap_or_default();
//             if sse_tx
//                 .send(Event::default().data(role_chunk))
//                 .await
//                 .is_err()
//             {
//                 #[cfg(feature = "trace")]
//                 tracing::warn!(chat_completion_id = %run.id, "sse receiver dropped before role chunk");
//                 return;
//             }
//
//             while let Some(ev) = run.rx.recv().await {
//                 match ev {
//                     Output(delta) => {
//                         #[cfg(feature = "trace")]
//                         {
//                             emitted_chunks += 1;
//                             tracing::trace!(
//                                 chat_completion_id = %run.id,
//                                 chunk = emitted_chunks,
//                                 chars = delta.text.chars().count(),
//                                 tokens = delta.tokens.len(),
//                                 "chat stream chunk"
//                             );
//                         }
//                         let chunk = run.stream_chunk(&delta, None);
//                         let json = sonic_rs::to_string(&chunk).unwrap_or_default();
//                         if sse_tx
//                             .send(Event::default().data(json))
//                             .await
//                             .is_err()
//                         {
//                             #[cfg(feature = "trace")]
//                             tracing::warn!(chat_completion_id = %run.id, "sse receiver dropped");
//                             break;
//                         }
//                     }
//                     Done(meta) => {
//                         #[cfg(feature = "trace")]
//                         tracing::info!(
//                             emitted_chunks,
//                             finish_reason = meta.reason.as_openai_str(),
//                             "chat completion stream done"
//                         );
//                         let final_chunk = run.finish_chunk(&meta);
//                         let final_json = sonic_rs::to_string(&final_chunk).unwrap_or_default();
//                         if sse_tx
//                             .send(Event::default().data(final_json))
//                             .await
//                             .is_err()
//                         {
//                             #[cfg(feature = "trace")]
//                             tracing::warn!(chat_completion_id = %run.id, "sse receiver dropped before final chunk");
//                             break;
//                         }
//                         let _ = sse_tx
//                             .send(Event::default().data("[DONE]"))
//                             .await;
//                         break;
//                     }
//                     Error(msg) => {
//                         #[cfg(feature = "trace")]
//                         tracing::error!(error = %msg, "chat completion stream error");
//                         let _ = sse_tx
//                             .send(Event::default().data(format!(
//                                 "{{\"error\":{}}}",
//                                 sonic_rs::to_string(&msg).unwrap_or_default()
//                             )))
//                             .await;
//                         break;
//                     }
//                 }
//             }
//             #[cfg(feature = "trace")]
//             tracing::info!(chat_completion_id = %run.id, emitted_chunks, "chat stream closed");
//         });
//         let sse_stream = ReceiverStream::new(sse_rx).map(Ok::<_, std::convert::Infallible>);
//
//         let keep_alive = KeepAlive::new().interval(Duration::from_millis(
//             app_state.runtime_manager.sse_keep_alive_ms(),
//         ));
//         return Sse::new(sse_stream)
//             .keep_alive(keep_alive)
//             .into_response();
//     }
//
//     match run.collect().await {
//         Ok(resp) => Json(resp).into_response(),
//         Err(e) => crate::access::http_api::infer_error_response(e),
//     }
// }
