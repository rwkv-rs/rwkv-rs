use axum::Json;

use crate::dtos::IndexResponse;

#[utoipa::path(
    get,
    path = "/",
    responses((status = 200, description = "API index", body = IndexResponse)),
    tag = "system"
)]
pub(crate) async fn index() -> Json<IndexResponse> {
    Json(IndexResponse {
        service: "rwkv-lm-eval-api",
        docs_url: "/openapi.json",
        openapi_url: "/openapi.json",
    })
}
