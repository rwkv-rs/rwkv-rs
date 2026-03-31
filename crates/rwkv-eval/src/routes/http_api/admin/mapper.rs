use crate::{
    dtos::{AdminDependencyResource, AdminHealthTargetResource},
    services::admin::{HealthTargetResult, RuntimeDependencyStatus},
};

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

pub(crate) fn to_admin_dependency_resource(
    dependency: RuntimeDependencyStatus,
) -> AdminDependencyResource {
    AdminDependencyResource {
        role: dependency.role.as_str().to_string(),
        label: dependency.label,
        base_url: dependency.base_url,
        status: dependency.status.as_str().to_string(),
        message: dependency.message,
        checked_at_unix_ms: dependency.checked_at_unix_ms,
    }
}
