use axum::Json;
use utoipa::OpenApi;

use crate::routes::http_api::openapi::ApiDoc;

#[utoipa::path(
    get,
    path = "/openapi.json",
    responses((status = 200, description = "OpenAPI specification")),
    tag = "system"
)]
pub(crate) async fn openapi_json() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}
