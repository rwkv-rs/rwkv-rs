//! `rwkv-infer` 按推理服务的职责拆分模块：
//! 路由层组织入口，处理中间件层承接请求，
//! 服务层负责编排推理业务与负载分发，核心层提供底层推理能力。

pub mod cores;
pub mod dtos;
pub mod handlers;
pub mod routes;
pub mod services;
