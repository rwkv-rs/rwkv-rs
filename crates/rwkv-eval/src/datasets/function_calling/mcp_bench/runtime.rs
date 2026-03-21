use super::types::{
    McpBenchAvailableTool, McpBenchItem, McpBenchPreflightSummary, McpBenchPreflightWire,
};
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use sonic_rs::{Value, json, prelude::*};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command as TokioCommand};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};

pub const MCP_BENCH_GIT_URL: &str = "https://github.com/Accenture/mcp-bench.git";
pub const MCP_BENCH_GIT_COMMIT: &str = "7a8eaeae83a842a2949080acc5473f65e1569daf";
pub const MCP_BENCH_EXPECTED_LEN: usize = 104;
pub const MCP_BENCH_TASK_FILES: [&str; 3] = [
    "mcpbench_tasks_single_runner_format.json",
    "mcpbench_tasks_multi_2server_runner_format.json",
    "mcpbench_tasks_multi_3server_runner_format.json",
];

pub(crate) const MCP_BENCH_HTTP_TIMEOUT_SECS: u64 = 60;
pub(crate) const MCP_BENCH_TOOL_DISCOVERY_TIMEOUT_SECS: u64 = 10;
pub(crate) const MCP_BENCH_SERVER_STARTUP_TIMEOUT_SECS: u64 = 30;
pub(crate) const MCP_BENCH_HEALTH_CHECK_TIMEOUT_SECS: u64 = 2;
pub(crate) const MCP_BENCH_PROCESS_WAIT_TIMEOUT_SECS: u64 = 5;
#[allow(dead_code)]
pub(crate) const MCP_BENCH_BATCH_TIMEOUT_SECS: u64 = 60;
pub(crate) const MCP_BENCH_DEFAULT_PORT: u16 = 3001;
pub(crate) const MCP_BENCH_PORT_SEARCH_ATTEMPTS: usize = 100;
pub(crate) const MCP_BENCH_RANDOM_PORT_MIN: u16 = 10000;
pub(crate) const MCP_BENCH_RANDOM_PORT_MAX: u16 = 50000;
pub(crate) const MCP_BENCH_TASK_TIMEOUT_SECS: u64 = 5000;
pub(crate) const MCP_BENCH_TASK_RETRY_MAX: usize = 3;
pub(crate) const MCP_BENCH_RETRY_DELAY_SECS: u64 = 5;
#[allow(dead_code)]
pub(crate) const MCP_BENCH_COMPRESSION_RETRIES: usize = 2;
#[allow(dead_code)]
pub(crate) const MCP_BENCH_SERVER_SEMAPHORE_LIMIT: usize = 20;
#[allow(dead_code)]
pub(crate) const MCP_BENCH_CONTENT_SUMMARY_THRESHOLD: usize = 1000;
pub(crate) const MCP_BENCH_CONTENT_TRUNCATE_LENGTH: usize = 4000;
pub(crate) const MCP_BENCH_ERROR_TRUNCATE_LENGTH: usize = 1000;
#[allow(dead_code)]
pub(crate) const MCP_BENCH_ERROR_DISPLAY_PREFIX: usize = 200;

const MCP_BENCH_PROTOCOL_VERSION: &str = "2024-11-05";
const MCP_BENCH_CLIENT_NAME: &str = "rwkv-eval-mcp-bench";
const MCP_BENCH_CLIENT_VERSION: &str = "1.0.0";
const MCP_BENCH_PYTHON_ENV_VAR: &str = "RWKV_MCP_BENCH_PYTHON";
const MCP_BENCH_VENV_DIR: &str = ".venv";
const MCP_BENCH_API_KEY_PLACEHOLDER: &str = "YOUR_KEY_HERE";
const MCP_BENCH_RESIDENT_SERVERS: [&str; 1] = ["Time MCP"];
const MCP_BENCH_PROBLEMATIC_TOOLS: [&str; 7] = [
    "Paper Search:search_semantic",
    "Paper Search:download_semantic",
    "Paper Search:read_semantic_paper",
    "OSINT Intelligence:osint_overview",
    "Paper Search:search_iacr",
    "Paper Search:read_iacr_paper",
    "Paper Search:download_iacr",
];

const PREFLIGHT_SCRIPT: &str = r###"
import asyncio
import contextlib
import io
import json
import os
import sys
import traceback

async def main() -> None:
    try:
        sys.path.insert(0, os.getcwd())
        from utils.collect_mcp_info import MCPServerInfoCollector

        collector = MCPServerInfoCollector("individual")
        configs = collector.load_server_configs()
        failures = []
        captured_stdout = []
        connected_servers = 0
        servers_with_tools = 0

        for config in configs:
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                info = await collector.test_individual_server(config)
            captured = stdout_buffer.getvalue().strip()
            if captured:
                captured_stdout.append(captured)

            status = info.get("connection_status", "")
            has_tools = bool(info.get("tools"))

            if status in ("success", "success_no_tools"):
                connected_servers += 1
            if has_tools:
                servers_with_tools += 1

            if status != "success" or not has_tools:
                failures.append({
                    "server_name": info.get("name", config.get("name", "")),
                    "connection_status": status,
                    "has_tools": has_tools,
                    "error": info.get("error", ""),
                })

        ok = connected_servers == len(configs) and servers_with_tools == len(configs)
        print(json.dumps({
            "ok": ok,
            "total_servers": len(configs),
            "connected_servers": connected_servers,
            "servers_with_tools": servers_with_tools,
            "failures": failures,
            "captured_stdout": "\n".join(captured_stdout),
        }, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }, ensure_ascii=False))

asyncio.run(main())
"###;

#[derive(Debug, Deserialize)]
struct RawCommandConfig {
    cmd: String,
    #[serde(default)]
    env: Vec<String>,
    #[serde(default)]
    cwd: String,
    #[serde(default)]
    transport: String,
    #[serde(default)]
    port: Option<u16>,
    #[serde(default)]
    endpoint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerTransport {
    Stdio,
    Http,
}

#[derive(Debug, Clone)]
struct ResolvedServerConfig {
    name: String,
    command: Vec<String>,
    cwd: PathBuf,
    env: BTreeMap<String, String>,
    transport: ServerTransport,
    configured_port: u16,
    endpoint: String,
}

pub(crate) struct McpBenchSession {
    connections: BTreeMap<String, McpServerConnection>,
    available_tools: BTreeMap<String, McpBenchAvailableTool>,
}

enum McpServerConnection {
    Stdio(StdioServerConnection),
    Http(HttpServerConnection),
}

struct StdioServerConnection {
    server_name: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_task: Option<JoinHandle<String>>,
    next_id: u64,
}

struct HttpServerConnection {
    server_name: String,
    child: Child,
    client: reqwest::Client,
    base_url: String,
    session_id: Option<String>,
    stderr_task: Option<JoinHandle<String>>,
    next_id: u64,
}

pub fn mcp_bench_root(base_dataset_root: &Path) -> PathBuf {
    base_dataset_root.join("mcp_bench")
}

pub fn runtime_root(base_dataset_root: &Path) -> PathBuf {
    mcp_bench_root(base_dataset_root).join("runtime")
}

pub fn task_assets_root(base_dataset_root: &Path) -> PathBuf {
    mcp_bench_root(base_dataset_root).join("tasks")
}

pub fn check_runtime_checkout(runtime_root: &Path) -> Result<bool, String> {
    if !runtime_root.join(".git").is_dir() {
        return Ok(false);
    }
    let output = StdCommand::new("git")
        .arg("-C")
        .arg(runtime_root)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .map_err(|err| format!("failed to inspect MCP-Bench checkout: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "git rev-parse failed for {}: {}",
            runtime_root.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim() == MCP_BENCH_GIT_COMMIT)
}

pub fn ensure_runtime_checkout(base_dataset_root: &Path) -> Result<(), String> {
    let bench_root = mcp_bench_root(base_dataset_root);
    let runtime_root = runtime_root(base_dataset_root);
    fs::create_dir_all(&bench_root)
        .map_err(|err| format!("failed to create {}: {err}", bench_root.display()))?;

    if runtime_root.exists() && !runtime_root.join(".git").is_dir() {
        return Err(format!(
            "mcp_bench runtime path exists but is not a git checkout: {}",
            runtime_root.display()
        ));
    }

    if !runtime_root.exists() {
        run_git(
            &bench_root,
            &["clone", MCP_BENCH_GIT_URL, "runtime"],
            "git clone mcp-bench runtime",
        )?;
    } else {
        run_git(
            &runtime_root,
            &["fetch", "--all", "--tags"],
            "git fetch mcp-bench runtime",
        )?;
    }

    run_git(
        &runtime_root,
        &["checkout", "--detach", MCP_BENCH_GIT_COMMIT],
        "git checkout pinned mcp-bench commit",
    )?;
    run_git(
        &runtime_root,
        &["submodule", "update", "--init", "--recursive"],
        "git submodule update mcp-bench runtime",
    )?;
    sync_task_assets(base_dataset_root)?;
    Ok(())
}

pub fn sync_task_assets(dataset_root: &Path) -> Result<(), String> {
    let tasks_root = task_assets_root(dataset_root);
    let runtime_tasks_root = runtime_root(dataset_root).join("tasks");
    fs::create_dir_all(&tasks_root)
        .map_err(|err| format!("failed to create {}: {err}", tasks_root.display()))?;

    for file_name in MCP_BENCH_TASK_FILES {
        let src = runtime_tasks_root.join(file_name);
        let dst = tasks_root.join(file_name);
        if !src.is_file() {
            return Err(format!(
                "missing official MCP-Bench task file: {}",
                src.display()
            ));
        }
        fs::copy(&src, &dst).map_err(|err| {
            format!(
                "failed to copy {} -> {}: {err}",
                src.display(),
                dst.display()
            )
        })?;
    }
    Ok(())
}

pub(crate) fn normalize_api_base(base_url: &str) -> String {
    let trimmed = base_url.trim();
    assert!(!trimmed.is_empty(), "base_url cannot be empty");
    let with_scheme = if trimmed.contains("://") {
        Cow::Borrowed(trimmed)
    } else {
        Cow::Owned(format!("http://{trimmed}"))
    };
    let without_trailing = with_scheme.trim_end_matches('/');
    if without_trailing.ends_with("/v1") {
        without_trailing.to_string()
    } else {
        format!("{without_trailing}/v1")
    }
}

pub async fn run_preflight(runtime_root: &Path) -> Result<McpBenchPreflightSummary, String> {
    let wire =
        run_python_wire::<McpBenchPreflightWire, ()>(runtime_root, PREFLIGHT_SCRIPT, None).await?;
    if !wire.ok {
        if !wire.error.is_empty() {
            return Err(wire.error);
        }

        let failures = wire
            .failures
            .iter()
            .map(|failure| {
                format!(
                    "{}(status={}, has_tools={}, error={})",
                    failure.server_name,
                    failure.connection_status,
                    failure.has_tools,
                    truncate_text(&failure.error, MCP_BENCH_ERROR_TRUNCATE_LENGTH)
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let captured_stdout = wire.captured_stdout.trim();
        let captured_stdout = if captured_stdout.is_empty() {
            String::new()
        } else {
            format!(
                "; captured_stdout={}",
                truncate_text(captured_stdout, MCP_BENCH_ERROR_TRUNCATE_LENGTH)
            )
        };
        return Err(format!(
            "MCP-Bench preflight failed: connected={}/{} returned_tools={}/{} failures=[{}]{}",
            wire.connected_servers,
            wire.total_servers,
            wire.servers_with_tools,
            wire.total_servers,
            failures,
            captured_stdout,
        ));
    }
    Ok(McpBenchPreflightSummary {
        total_servers: wire.total_servers,
        connected_servers: wire.connected_servers,
        servers_with_tools: wire.servers_with_tools,
        failures: wire.failures,
        captured_stdout: wire.captured_stdout,
    })
}

impl McpBenchSession {
    pub(crate) async fn connect(runtime_root: &Path, item: &McpBenchItem) -> Result<Self, String> {
        let command_configs = load_command_configs(runtime_root)?;
        let api_keys = load_api_keys(runtime_root)?;

        let mut session = Self {
            connections: BTreeMap::new(),
            available_tools: BTreeMap::new(),
        };

        for server_name in collect_task_server_names(item) {
            let resolved =
                resolve_server_config(runtime_root, &server_name, &command_configs, &api_keys)?;
            let mut connection = McpServerConnection::connect(&resolved).await?;
            let discovered = connection.discover_tools().await?;

            for tool in discovered {
                if MCP_BENCH_PROBLEMATIC_TOOLS.contains(&tool.name.as_str()) {
                    continue;
                }
                session.available_tools.insert(tool.name.clone(), tool);
            }

            session.connections.insert(server_name, connection);
        }

        Ok(session)
    }

    pub(crate) fn available_tools(&self) -> &BTreeMap<String, McpBenchAvailableTool> {
        &self.available_tools
    }

    pub(crate) async fn call_tool(
        &mut self,
        full_tool_name: &str,
        arguments: &Value,
    ) -> Result<(bool, String), String> {
        let (server_name, tool_name) = full_tool_name
            .split_once(':')
            .ok_or_else(|| format!("invalid MCP tool name `{full_tool_name}`"))?;
        let connection = self
            .connections
            .get_mut(server_name)
            .ok_or_else(|| format!("no active MCP connection for server `{server_name}`"))?;
        connection.call_tool(tool_name, arguments).await
    }

    pub(crate) async fn shutdown(self) {
        for connection in self.connections.into_values() {
            connection.shutdown().await;
        }
    }
}

impl McpServerConnection {
    async fn connect(config: &ResolvedServerConfig) -> Result<Self, String> {
        match config.transport {
            ServerTransport::Stdio => Ok(Self::Stdio(StdioServerConnection::start(config).await?)),
            ServerTransport::Http => Ok(Self::Http(HttpServerConnection::start(config).await?)),
        }
    }

    async fn discover_tools(&mut self) -> Result<Vec<McpBenchAvailableTool>, String> {
        match self {
            Self::Stdio(connection) => connection.discover_tools().await,
            Self::Http(connection) => connection.discover_tools().await,
        }
    }

    async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<(bool, String), String> {
        match self {
            Self::Stdio(connection) => connection.call_tool(tool_name, arguments).await,
            Self::Http(connection) => connection.call_tool(tool_name, arguments).await,
        }
    }

    async fn shutdown(self) {
        match self {
            Self::Stdio(connection) => connection.shutdown().await,
            Self::Http(connection) => connection.shutdown().await,
        }
    }
}

impl StdioServerConnection {
    async fn start(config: &ResolvedServerConfig) -> Result<Self, String> {
        let mut command = TokioCommand::new(&config.command[0]);
        if config.command.len() > 1 {
            command.args(&config.command[1..]);
        }
        command
            .current_dir(&config.cwd)
            .envs(config.env.clone())
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|err| {
            format!(
                "failed to spawn MCP stdio server `{}` in {}: {err}",
                config.name,
                config.cwd.display()
            )
        })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| format!("failed to open stdin for MCP server `{}`", config.name))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| format!("failed to open stdout for MCP server `{}`", config.name))?;
        let stderr_task = child.stderr.take().map(spawn_stderr_collector);

        let mut connection = Self {
            server_name: config.name.clone(),
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr_task,
            next_id: 1,
        };

        if let Err(err) = connection.initialize().await {
            connection.shutdown().await;
            return Err(err);
        }
        Ok(connection)
    }

    async fn initialize(&mut self) -> Result<(), String> {
        let init_params = json!({
            "protocolVersion": MCP_BENCH_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {
                "name": MCP_BENCH_CLIENT_NAME,
                "version": MCP_BENCH_CLIENT_VERSION,
            },
        });
        self.request("initialize", Some(init_params)).await?;
        self.notify("notifications/initialized", Some(json!({})))
            .await?;
        Ok(())
    }

    async fn discover_tools(&mut self) -> Result<Vec<McpBenchAvailableTool>, String> {
        let response = self.request("tools/list", Some(json!({}))).await?;
        parse_discovered_tools(&self.server_name, &response)
    }

    async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<(bool, String), String> {
        let response = self
            .request(
                "tools/call",
                Some(json!({
                    "name": tool_name,
                    "arguments": arguments,
                })),
            )
            .await?;
        let is_error = response
            .get("isError")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        Ok((!is_error, extract_text_from_result(&response)))
    }

    async fn request(&mut self, method: &str, params: Option<Value>) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;

        let payload = if let Some(params) = params {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            })
        } else {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
            })
        };

        write_stdio_message(&mut self.stdin, &payload)
            .await
            .map_err(|err| format!("failed to send `{method}` to `{}`: {err}", self.server_name))?;

        loop {
            let message = read_stdio_message(&mut self.stdout).await.map_err(|err| {
                format!(
                    "failed to read response from MCP server `{}` during `{method}`: {err}",
                    self.server_name
                )
            })?;

            if message
                .get("id")
                .and_then(|value| value.as_u64())
                .is_some_and(|resp_id| resp_id == id)
            {
                if let Some(error) = message.get("error") {
                    return Err(format_json_rpc_error(
                        &self.server_name,
                        method,
                        error,
                        self.stderr_task.as_ref(),
                    ));
                }
                return Ok(message
                    .get("result")
                    .cloned()
                    .unwrap_or_else(|| json!(null)));
            }
        }
    }

    async fn notify(&mut self, method: &str, params: Option<Value>) -> Result<(), String> {
        let payload = if let Some(params) = params {
            json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            })
        } else {
            json!({
                "jsonrpc": "2.0",
                "method": method,
            })
        };
        write_stdio_message(&mut self.stdin, &payload)
            .await
            .map_err(|err| format!("failed to send notification `{method}`: {err}"))
    }

    async fn shutdown(mut self) {
        let _ = self.stdin.shutdown().await;
        let _ = self.child.kill().await;
        let _ = timeout(
            Duration::from_secs(MCP_BENCH_PROCESS_WAIT_TIMEOUT_SECS),
            self.child.wait(),
        )
        .await;
        if let Some(stderr_task) = self.stderr_task.take() {
            let _ = stderr_task.await;
        }
    }
}

impl HttpServerConnection {
    async fn start(config: &ResolvedServerConfig) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(MCP_BENCH_HTTP_TIMEOUT_SECS))
            .build()
            .map_err(|err| format!("failed to build HTTP client for `{}`: {err}", config.name))?;

        let mut last_error = String::new();
        for port in candidate_ports(config.configured_port) {
            match Self::start_with_port(config, client.clone(), port).await {
                Ok(connection) => return Ok(connection),
                Err(err) => last_error = err,
            }
        }

        Err(format!(
            "failed to start HTTP MCP server `{}` after {} port attempts: {}",
            config.name, MCP_BENCH_PORT_SEARCH_ATTEMPTS, last_error
        ))
    }

    async fn start_with_port(
        config: &ResolvedServerConfig,
        client: reqwest::Client,
        port: u16,
    ) -> Result<Self, String> {
        let mut env = config.env.clone();
        env.insert("MCP_SERVER_PORT".to_string(), port.to_string());

        let command_parts = rewrite_http_command(&config.command, config.configured_port, port);
        let mut command = TokioCommand::new(&command_parts[0]);
        if command_parts.len() > 1 {
            command.args(&command_parts[1..]);
        }
        command
            .current_dir(&config.cwd)
            .envs(env)
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|err| {
            format!(
                "failed to spawn HTTP MCP server `{}` in {}: {err}",
                config.name,
                config.cwd.display()
            )
        })?;
        let stderr_task = child.stderr.take().map(spawn_stderr_collector);
        let base_url = format!("http://localhost:{}{}", port, config.endpoint);

        let startup_result = wait_for_http_server(&mut child, &client, &base_url).await;
        if let Err(err) = startup_result {
            let stderr = take_stderr(stderr_task).await;
            let _ = child.kill().await;
            let _ = timeout(
                Duration::from_secs(MCP_BENCH_PROCESS_WAIT_TIMEOUT_SECS),
                child.wait(),
            )
            .await;
            return Err(if stderr.trim().is_empty() {
                format!("failed to start HTTP MCP server `{}`: {err}", config.name)
            } else {
                format!(
                    "failed to start HTTP MCP server `{}`: {err}; stderr={}",
                    config.name,
                    truncate_text(&stderr, MCP_BENCH_ERROR_TRUNCATE_LENGTH)
                )
            });
        }

        Ok(Self {
            server_name: config.name.clone(),
            child,
            client,
            base_url,
            session_id: None,
            stderr_task,
            next_id: 1,
        })
    }

    async fn discover_tools(&mut self) -> Result<Vec<McpBenchAvailableTool>, String> {
        let init_params = json!({
            "protocolVersion": MCP_BENCH_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "clientInfo": {
                "name": MCP_BENCH_CLIENT_NAME,
                "version": MCP_BENCH_CLIENT_VERSION,
            },
        });
        self.request("initialize", Some(init_params)).await?;
        self.notify("notifications/initialized", None).await?;
        let response = self.request("tools/list", Some(json!({}))).await?;
        parse_discovered_tools(&self.server_name, &response)
    }

    async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: &Value,
    ) -> Result<(bool, String), String> {
        let response = self
            .request(
                "tools/call",
                Some(json!({
                    "name": tool_name,
                    "arguments": arguments,
                })),
            )
            .await?;
        let is_error = response
            .get("isError")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        Ok((!is_error, extract_text_from_result(&response)))
    }

    async fn request(&mut self, method: &str, params: Option<Value>) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;

        let payload = if let Some(params) = params {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params,
            })
        } else {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
            })
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        if let Some(session_id) = &self.session_id {
            let value = HeaderValue::from_str(session_id)
                .map_err(|err| format!("invalid MCP session id header: {err}"))?;
            headers.insert("mcp-session-id", value);
        }

        let response = self
            .client
            .post(&self.base_url)
            .headers(headers)
            .body(
                sonic_rs::to_string(&payload)
                    .map_err(|err| format!("failed to encode HTTP MCP request json: {err}"))?,
            )
            .send()
            .await
            .map_err(|err| {
                format!(
                    "HTTP MCP request `{method}` to `{}` failed: {err}",
                    self.server_name
                )
            })?;

        if let Some(session_id) = response.headers().get("mcp-session-id") {
            if let Ok(session_id) = session_id.to_str() {
                self.session_id = Some(session_id.to_string());
            }
        }

        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let body = response.text().await.map_err(|err| {
            format!(
                "failed to read HTTP MCP response body from `{}`: {err}",
                self.server_name
            )
        })?;
        if !status.is_success() {
            return Err(format!(
                "HTTP MCP request `{method}` to `{}` returned {}: {}",
                self.server_name,
                status,
                truncate_text(&body, MCP_BENCH_ERROR_TRUNCATE_LENGTH)
            ));
        }

        let message = parse_http_response_body(&body, &content_type).map_err(|err| {
            format!(
                "failed to parse HTTP MCP response for `{}` `{method}`: {err}",
                self.server_name
            )
        })?;
        if let Some(error) = message.get("error") {
            return Err(format_json_rpc_error(
                &self.server_name,
                method,
                error,
                self.stderr_task.as_ref(),
            ));
        }
        Ok(message
            .get("result")
            .cloned()
            .unwrap_or_else(|| json!(null)))
    }

    async fn notify(&mut self, method: &str, params: Option<Value>) -> Result<(), String> {
        let payload = if let Some(params) = params {
            json!({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            })
        } else {
            json!({
                "jsonrpc": "2.0",
                "method": method,
            })
        };

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/json, text/event-stream"),
        );
        if let Some(session_id) = &self.session_id {
            let value = HeaderValue::from_str(session_id)
                .map_err(|err| format!("invalid MCP session id header: {err}"))?;
            headers.insert("mcp-session-id", value);
        }

        let response = self
            .client
            .post(&self.base_url)
            .headers(headers)
            .body(
                sonic_rs::to_string(&payload)
                    .map_err(|err| format!("failed to encode HTTP MCP notification json: {err}"))?,
            )
            .send()
            .await
            .map_err(|err| {
                format!(
                    "HTTP MCP notification `{method}` to `{}` failed: {err}",
                    self.server_name
                )
            })?;

        if let Some(session_id) = response.headers().get("mcp-session-id") {
            if let Ok(session_id) = session_id.to_str() {
                self.session_id = Some(session_id.to_string());
            }
        }

        let status = response.status();
        let body = response.text().await.map_err(|err| {
            format!(
                "failed to read HTTP MCP notification response body from `{}`: {err}",
                self.server_name
            )
        })?;
        if !status.is_success() {
            return Err(format!(
                "HTTP MCP notification `{method}` to `{}` returned {}: {}",
                self.server_name,
                status,
                truncate_text(&body, MCP_BENCH_ERROR_TRUNCATE_LENGTH)
            ));
        }

        Ok(())
    }

    async fn shutdown(mut self) {
        let _ = self.child.kill().await;
        let _ = timeout(
            Duration::from_secs(MCP_BENCH_PROCESS_WAIT_TIMEOUT_SECS),
            self.child.wait(),
        )
        .await;
        if let Some(stderr_task) = self.stderr_task.take() {
            let _ = stderr_task.await;
        }
    }
}

fn collect_task_server_names(item: &McpBenchItem) -> Vec<String> {
    let mut ordered = Vec::new();
    let mut seen = BTreeSet::new();

    let required_servers = if item.servers.is_empty() {
        item.server_name
            .split('+')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>()
    } else {
        item.servers.clone()
    };

    for server_name in required_servers
        .into_iter()
        .chain(
            MCP_BENCH_RESIDENT_SERVERS
                .iter()
                .map(|name| name.to_string()),
        )
        .chain(item.task.distraction_servers.iter().cloned())
    {
        if seen.insert(server_name.clone()) {
            ordered.push(server_name);
        }
    }

    ordered
}

fn load_command_configs(runtime_root: &Path) -> Result<BTreeMap<String, RawCommandConfig>, String> {
    let path = runtime_root.join("mcp_servers").join("commands.json");
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    sonic_rs::from_str(&text).map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn load_api_keys(runtime_root: &Path) -> Result<BTreeMap<String, String>, String> {
    let path = runtime_root.join("mcp_servers").join("api_key");
    if !path.exists() {
        return Ok(BTreeMap::new());
    }

    let text = fs::read_to_string(&path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut api_keys = BTreeMap::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() || value == MCP_BENCH_API_KEY_PLACEHOLDER {
            continue;
        }
        api_keys.insert(key.trim().to_string(), value.to_string());
    }
    Ok(api_keys)
}

fn rewrite_python_command(runtime_root: &Path, command: &mut Vec<String>) {
    if let Some(rewritten) = rewrite_uv_run_command(runtime_root, command) {
        *command = rewritten;
        return;
    }

    if let Some(program) = command.first_mut()
        && matches!(program.as_str(), "python" | "python3")
    {
        *program = resolve_python_command(runtime_root).display().to_string();
    }
}

fn rewrite_uv_run_command(runtime_root: &Path, command: &[String]) -> Option<Vec<String>> {
    if command.len() < 3 || command[0] != "uv" || command[1] != "run" {
        return None;
    }

    if matches!(command[2].as_str(), "python" | "python3") {
        let mut rewritten = vec![resolve_python_command(runtime_root).display().to_string()];
        rewritten.extend(command[3..].iter().cloned());
        return Some(rewritten);
    }

    let bin_dir = resolve_python_command(runtime_root)
        .parent()
        .map(Path::to_path_buf)?;
    let candidate = bin_dir.join(&command[2]);
    if !candidate.is_file() {
        return None;
    }

    let mut rewritten = vec![candidate.display().to_string()];
    rewritten.extend(command[3..].iter().cloned());
    Some(rewritten)
}

fn resolve_python_command(runtime_root: &Path) -> PathBuf {
    if let Ok(value) = env::var(MCP_BENCH_PYTHON_ENV_VAR) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }

    let venv_bin = runtime_root.join(MCP_BENCH_VENV_DIR).join("bin");
    let python3 = venv_bin.join("python3");
    if python3.is_file() {
        return python3;
    }

    let python = venv_bin.join("python");
    if python.is_file() {
        return python;
    }

    PathBuf::from("python3")
}

fn extend_python_runtime_env(runtime_root: &Path, env_map: &mut BTreeMap<String, String>) {
    let python = resolve_python_command(runtime_root);
    let Some(bin_dir) = python.parent() else {
        return;
    };
    let Some(venv_dir) = bin_dir.parent() else {
        return;
    };
    if !bin_dir.ends_with("bin") || !venv_dir.exists() {
        return;
    }

    let existing_path = env_map
        .get("PATH")
        .cloned()
        .or_else(|| env::var("PATH").ok())
        .unwrap_or_default();
    let mut paths = vec![bin_dir.to_path_buf()];
    paths.extend(env::split_paths(&OsString::from(existing_path)));

    let combined_path = env::join_paths(paths)
        .ok()
        .and_then(|joined| joined.into_string().ok())
        .unwrap_or_else(|| bin_dir.display().to_string());
    env_map.insert("PATH".to_string(), combined_path);
    env_map.insert(
        "VIRTUAL_ENV".to_string(),
        venv_dir.as_os_str().to_string_lossy().into_owned(),
    );
}

fn resolve_server_config(
    runtime_root: &Path,
    server_name: &str,
    command_configs: &BTreeMap<String, RawCommandConfig>,
    api_keys: &BTreeMap<String, String>,
) -> Result<ResolvedServerConfig, String> {
    let raw = command_configs
        .get(server_name)
        .ok_or_else(|| format!("no MCP command configuration found for server `{server_name}`"))?;

    let mut command = raw
        .cmd
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if command.is_empty() {
        return Err(format!("empty MCP command for server `{server_name}`"));
    }
    rewrite_python_command(runtime_root, &mut command);

    let cwd = if raw.cwd.trim().is_empty() {
        runtime_root.to_path_buf()
    } else if let Some(relative) = raw.cwd.strip_prefix("../") {
        runtime_root.join("mcp_servers").join(relative)
    } else {
        runtime_root.join(&raw.cwd)
    };

    let mut env = BTreeMap::new();
    for key in &raw.env {
        if let Some(value) = api_keys.get(key) {
            env.insert(key.clone(), value.clone());
        } else if let Ok(value) = std::env::var(key) {
            env.insert(key.clone(), value);
        }
    }
    extend_python_runtime_env(runtime_root, &mut env);

    let transport = if raw.transport.eq_ignore_ascii_case("http") {
        ServerTransport::Http
    } else {
        ServerTransport::Stdio
    };

    Ok(ResolvedServerConfig {
        name: server_name.to_string(),
        command,
        cwd,
        env,
        transport,
        configured_port: raw.port.unwrap_or(MCP_BENCH_DEFAULT_PORT),
        endpoint: raw.endpoint.clone().unwrap_or_else(|| "/mcp".to_string()),
    })
}

async fn write_stdio_message(stdin: &mut ChildStdin, payload: &Value) -> Result<(), String> {
    let body =
        sonic_rs::to_string(payload).map_err(|err| format!("failed to encode jsonrpc: {err}"))?;
    stdin
        .write_all(body.as_bytes())
        .await
        .map_err(|err| format!("failed to write jsonrpc body: {err}"))?;
    stdin
        .write_all(b"\n")
        .await
        .map_err(|err| format!("failed to write jsonrpc newline delimiter: {err}"))?;
    stdin
        .flush()
        .await
        .map_err(|err| format!("failed to flush jsonrpc body: {err}"))
}

async fn read_stdio_message(stdout: &mut BufReader<ChildStdout>) -> Result<Value, String> {
    loop {
        let mut line = String::new();
        let bytes_read = timeout(
            Duration::from_secs(MCP_BENCH_TOOL_DISCOVERY_TIMEOUT_SECS),
            stdout.read_line(&mut line),
        )
        .await
        .map_err(|_| "timed out waiting for MCP stdio header".to_string())?
        .map_err(|err| format!("failed to read MCP stdio header: {err}"))?;
        if bytes_read == 0 {
            return Err("MCP stdio stream closed while reading headers".to_string());
        }

        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return sonic_rs::from_str(trimmed).map_err(|err| {
                format!(
                    "invalid newline-delimited MCP stdio jsonrpc payload: {err}; body={trimmed:?}"
                )
            });
        }

        let mut content_length = if let Some((name, value)) = trimmed.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|err| format!("invalid Content-Length header `{trimmed}`: {err}"))?,
            )
        } else {
            continue;
        };

        loop {
            let mut header_line = String::new();
            let bytes_read = timeout(
                Duration::from_secs(MCP_BENCH_TOOL_DISCOVERY_TIMEOUT_SECS),
                stdout.read_line(&mut header_line),
            )
            .await
            .map_err(|_| "timed out waiting for MCP stdio header".to_string())?
            .map_err(|err| format!("failed to read MCP stdio header: {err}"))?;
            if bytes_read == 0 {
                return Err("MCP stdio stream closed while reading headers".to_string());
            }

            let trimmed = header_line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }

            if let Some((name, value)) = trimmed.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                content_length =
                    Some(value.trim().parse::<usize>().map_err(|err| {
                        format!("invalid Content-Length header `{trimmed}`: {err}")
                    })?);
            }
        }

        let content_length = content_length
            .ok_or_else(|| "missing Content-Length header in MCP stdio response".to_string())?;
        let mut body = vec![0_u8; content_length];
        timeout(
            Duration::from_secs(MCP_BENCH_HTTP_TIMEOUT_SECS),
            stdout.read_exact(&mut body),
        )
        .await
        .map_err(|_| "timed out waiting for MCP stdio body".to_string())?
        .map_err(|err| format!("failed to read MCP stdio body: {err}"))?;

        let body = String::from_utf8(body)
            .map_err(|err| format!("MCP stdio body was not valid utf-8: {err}"))?;
        return sonic_rs::from_str(&body)
            .map_err(|err| format!("invalid MCP stdio jsonrpc payload: {err}; body={body:?}"));
    }
}

fn parse_discovered_tools(
    server_name: &str,
    response: &Value,
) -> Result<Vec<McpBenchAvailableTool>, String> {
    let tools = response
        .get("tools")
        .and_then(|value| value.as_array())
        .ok_or_else(|| format!("`tools/list` response from `{server_name}` was missing `tools`"))?;

    let mut parsed = Vec::new();
    for tool in tools {
        let Some(tool_name) = tool.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        let full_name = format!("{server_name}:{tool_name}");
        parsed.push(McpBenchAvailableTool {
            name: full_name,
            original_name: tool_name.to_string(),
            server: server_name.to_string(),
            description: tool
                .get("description")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string(),
            input_schema: tool
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({})),
        });
    }
    Ok(parsed)
}

fn extract_text_from_result(result: &Value) -> String {
    if let Some(content_items) = result.get("content").and_then(|value| value.as_array()) {
        let texts = content_items
            .iter()
            .filter_map(|item| item.get("text").and_then(|value| value.as_str()))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>();
        if !texts.is_empty() {
            return truncate_text(&texts.join("\n"), MCP_BENCH_CONTENT_TRUNCATE_LENGTH);
        }
    }

    if let Some(text) = result.as_str() {
        return truncate_text(text, MCP_BENCH_CONTENT_TRUNCATE_LENGTH);
    }

    sonic_rs::to_string_pretty(result)
        .map(|text| truncate_text(&text, MCP_BENCH_CONTENT_TRUNCATE_LENGTH))
        .unwrap_or_else(|_| "tool call returned an unreadable JSON payload".to_string())
}

async fn wait_for_http_server(
    child: &mut Child,
    client: &reqwest::Client,
    base_url: &str,
) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(MCP_BENCH_SERVER_STARTUP_TIMEOUT_SECS);

    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|err| format!("failed to poll child status: {err}"))?
        {
            return Err(format!("process exited early with status {status}"));
        }

        let current_health_error = match timeout(
            Duration::from_secs(MCP_BENCH_HEALTH_CHECK_TIMEOUT_SECS),
            client.get(base_url).send(),
        )
        .await
        {
            Ok(Ok(_)) => return Ok(()),
            Ok(Err(err)) => err.to_string(),
            Err(_) => "health check timed out".to_string(),
        };

        if Instant::now() >= deadline {
            if child
                .try_wait()
                .map_err(|err| format!("failed to poll child status: {err}"))?
                .is_none()
            {
                return Ok(());
            }
            return Err(current_health_error);
        }

        sleep(Duration::from_millis(500)).await;
    }
}

fn rewrite_http_command(command: &[String], configured_port: u16, new_port: u16) -> Vec<String> {
    let mut updated = Vec::with_capacity(command.len());
    let mut index = 0;
    while index < command.len() {
        let part = &command[index];
        if part == "--port" && index + 1 < command.len() {
            updated.push(part.clone());
            updated.push(new_port.to_string());
            index += 2;
            continue;
        }
        if let Some(prefix) = part.strip_prefix("--port=") {
            if prefix == configured_port.to_string() {
                updated.push(format!("--port={new_port}"));
            } else {
                updated.push(part.clone());
            }
            index += 1;
            continue;
        }
        updated.push(part.clone());
        index += 1;
    }
    updated
}

fn candidate_ports(configured_port: u16) -> Vec<u16> {
    let mut ports = Vec::new();
    if port_is_available(configured_port) {
        ports.push(configured_port);
    } else {
        ports.push(configured_port);
    }

    let range = (MCP_BENCH_RANDOM_PORT_MAX - MCP_BENCH_RANDOM_PORT_MIN + 1) as u64;
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
        % range;

    for attempt in 0..MCP_BENCH_PORT_SEARCH_ATTEMPTS {
        let port = MCP_BENCH_RANDOM_PORT_MIN + ((seed + attempt as u64) % range) as u16;
        if !ports.contains(&port) {
            ports.push(port);
        }
    }
    ports
}

fn port_is_available(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

fn parse_http_response_body(body: &str, content_type: &str) -> Result<Value, String> {
    if content_type.contains("text/event-stream") {
        for line in body.lines() {
            if let Some(payload) = line.strip_prefix("data: ") {
                if let Ok(value) = sonic_rs::from_str::<Value>(payload) {
                    return Ok(value);
                }
            }
        }
        return Err(format!(
            "no JSON data event found in SSE response: {body:?}"
        ));
    }
    sonic_rs::from_str(body).map_err(|err| format!("invalid JSON response: {err}; body={body:?}"))
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= max_chars {
        return trimmed.to_string();
    }
    trimmed.chars().take(max_chars).collect::<String>() + "..."
}

fn format_json_rpc_error(
    server_name: &str,
    method: &str,
    error: &Value,
    stderr_task: Option<&JoinHandle<String>>,
) -> String {
    let code = error
        .get("code")
        .and_then(|value| value.as_i64())
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = error
        .get("message")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown JSON-RPC error");
    let mut rendered =
        format!("MCP call `{method}` on server `{server_name}` failed with code {code}: {message}");
    if let Some(data) = error.get("data") {
        rendered.push_str(&format!(
            "; data={}",
            truncate_text(
                &sonic_rs::to_string(data).unwrap_or_else(|_| "<unrenderable>".to_string()),
                MCP_BENCH_ERROR_TRUNCATE_LENGTH
            )
        ));
    }
    if let Some(stderr_task) = stderr_task
        && stderr_task.is_finished()
    {
        rendered.push_str("; server stderr buffered");
    }
    rendered
}

fn spawn_stderr_collector(stderr: ChildStderr) -> JoinHandle<String> {
    tokio::spawn(async move {
        let mut stderr = BufReader::new(stderr);
        let mut buffer = String::new();
        let _ = stderr.read_to_string(&mut buffer).await;
        buffer
    })
}

async fn take_stderr(stderr_task: Option<JoinHandle<String>>) -> String {
    match stderr_task {
        Some(task) => task.await.unwrap_or_default(),
        None => String::new(),
    }
}

pub(crate) async fn run_python_wire<TResp, TReq>(
    runtime_root: &Path,
    script: &str,
    request: Option<&TReq>,
) -> Result<TResp, String>
where
    TResp: serde::de::DeserializeOwned,
    TReq: Serialize,
{
    let python = resolve_python_command(runtime_root);
    let mut runtime_env = BTreeMap::new();
    extend_python_runtime_env(runtime_root, &mut runtime_env);

    let mut child = TokioCommand::new(&python)
        .arg("-c")
        .arg(script)
        .current_dir(runtime_root)
        .envs(runtime_env)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|err| {
            format!(
                "failed to spawn MCP-Bench python `{}` under {}: {err}",
                python.display(),
                runtime_root.display()
            )
        })?;

    if let Some(request) = request {
        let request_json = sonic_rs::to_string(request)
            .map_err(|err| format!("failed to encode request json: {err}"))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "failed to open python stdin".to_string())?;
        stdin
            .write_all(request_json.as_bytes())
            .await
            .map_err(|err| format!("failed to write python stdin: {err}"))?;
        stdin
            .shutdown()
            .await
            .map_err(|err| format!("failed to close python stdin: {err}"))?;
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|err| format!("failed to wait for python wrapper: {err}"))?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|err| format!("python stdout was not valid utf-8: {err}"))?;
    let stderr = String::from_utf8(output.stderr)
        .map_err(|err| format!("python stderr was not valid utf-8: {err}"))?;

    if !output.status.success() {
        return Err(format!(
            "python wrapper exited with status {} under {}: stderr={}",
            output.status,
            runtime_root.display(),
            stderr.trim()
        ));
    }

    let stdout = stdout.trim();
    if stdout.is_empty() {
        return Err(format!(
            "python wrapper returned empty stdout under {}: stderr={}",
            runtime_root.display(),
            stderr.trim()
        ));
    }

    sonic_rs::from_str(stdout).map_err(|err| {
        format!(
            "failed to parse python wrapper json under {}: {err}; stdout={stdout:?}; stderr={}",
            runtime_root.display(),
            stderr.trim()
        )
    })
}

fn run_git(cwd: &Path, args: &[&str], label: &str) -> Result<(), String> {
    let output = StdCommand::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|err| format!("{label} failed to start in {}: {err}", cwd.display()))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{label} failed in {}: status={} stderr={}",
        cwd.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        MCP_BENCH_DEFAULT_PORT, MCP_BENCH_TASK_RETRY_MAX, normalize_api_base, rewrite_http_command,
    };

    #[test]
    fn normalize_api_base_adds_scheme_and_v1() {
        assert_eq!(
            normalize_api_base("127.0.0.1:8080"),
            "http://127.0.0.1:8080/v1"
        );
        assert_eq!(
            normalize_api_base("https://example.com/v1/"),
            "https://example.com/v1"
        );
    }

    #[test]
    fn rewrite_http_command_replaces_port_flag() {
        let command = vec![
            "node".to_string(),
            "dist/cli.js".to_string(),
            "--port".to_string(),
            MCP_BENCH_DEFAULT_PORT.to_string(),
        ];
        assert_eq!(
            rewrite_http_command(&command, MCP_BENCH_DEFAULT_PORT, 31337),
            vec![
                "node".to_string(),
                "dist/cli.js".to_string(),
                "--port".to_string(),
                "31337".to_string(),
            ]
        );
        assert_eq!(MCP_BENCH_TASK_RETRY_MAX, 3);
    }
}
