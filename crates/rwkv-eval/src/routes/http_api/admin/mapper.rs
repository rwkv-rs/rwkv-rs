use crate::{dtos::AdminHealthTargetResource, services::admin::HealthTargetResult};

pub(crate) fn to_admin_health_target_resource(
    target: HealthTargetResult,
) -> AdminHealthTargetResource {
    AdminHealthTargetResource {
        base_url: target.base_url,
        roles: target.roles,
        status: target.status,
        error: target.error,
        health: target.health,
    }
}
