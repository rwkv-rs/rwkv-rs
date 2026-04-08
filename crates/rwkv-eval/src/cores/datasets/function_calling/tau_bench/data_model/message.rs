use crate::cores::datasets::function_calling::{FunctionCall, FunctionCallingStep, ToolRequestor};

use super::tasks::{RawMessage, TauTask};

#[derive(Debug, Clone)]
pub struct SystemMessage {
    pub role: &'static str,
    pub content: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: sonic_rs::Object,
    pub requestor: ToolRequestor,
}

#[derive(Debug, Clone)]
pub struct ParticipantMessageBase {
    pub role: &'static str,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ParticipantMessageBase {
    pub fn has_text_content(&self) -> bool {
        self.content
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
    }

    pub fn is_tool_call(&self) -> bool {
        self.tool_calls.is_some()
    }
}

pub type AssistantMessage = ParticipantMessageBase;
pub type UserMessage = ParticipantMessageBase;

#[derive(Debug, Clone)]
pub struct ToolMessage {
    pub id: String,
    pub role: &'static str,
    pub content: Option<String>,
    pub requestor: ToolRequestor,
    pub error: bool,
}

#[derive(Debug, Clone)]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolMessage),
}

impl Message {
    pub fn role(&self) -> &'static str {
        match self {
            Self::System(message) => message.role,
            Self::User(message) => message.role,
            Self::Assistant(message) => message.role,
            Self::Tool(message) => message.role,
        }
    }

    pub fn content(&self) -> Option<&str> {
        match self {
            Self::System(message) => message.content.as_deref(),
            Self::User(message) => message.content.as_deref(),
            Self::Assistant(message) => message.content.as_deref(),
            Self::Tool(message) => message.content.as_deref(),
        }
    }

    pub fn assistant_message(&self) -> Option<&AssistantMessage> {
        match self {
            Self::Assistant(message) => Some(message),
            _ => None,
        }
    }

    pub fn participant_message(&self) -> Option<&ParticipantMessageBase> {
        match self {
            Self::User(message) | Self::Assistant(message) => Some(message),
            _ => None,
        }
    }

    pub fn to_nl_assertion_line(&self) -> String {
        match self {
            Self::System(message) => format!("system: {}", message.content.as_deref().unwrap_or("")),
            Self::User(message) => render_participant_line("user", message),
            Self::Assistant(message) => render_participant_line("assistant", message),
            Self::Tool(message) => format!(
                "tool: id={} requestor={} error={} content={}",
                message.id,
                message.requestor.as_str(),
                message.error,
                message.content.as_deref().unwrap_or("")
            ),
        }
    }
}

fn render_participant_line(role: &str, message: &ParticipantMessageBase) -> String {
    if let Some(tool_calls) = &message.tool_calls {
        let rendered = tool_calls
            .iter()
            .map(|tool_call| {
                format!(
                    "{{\"id\":\"{}\",\"requestor\":\"{}\",\"name\":\"{}\",\"arguments\":{}}}",
                    tool_call.id,
                    tool_call.requestor.as_str(),
                    tool_call.name,
                    sonic_rs::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_string())
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("{role}: tool_calls=[{rendered}]")
    } else {
        format!("{role}: {}", message.content.as_deref().unwrap_or(""))
    }
}

pub fn build_full_trajectory(
    task: &TauTask,
    system_prompt: &str,
    user_prompt: &str,
    steps: &[FunctionCallingStep],
    final_answer: Option<&str>,
) -> Vec<Message> {
    let mut full_trajectory = vec![
        Message::System(SystemMessage {
            role: "system",
            content: Some(system_prompt.to_string()),
        }),
        Message::User(UserMessage {
            role: "user",
            content: Some(user_prompt.to_string()),
            tool_calls: None,
        }),
    ];

    full_trajectory.extend(
        task
        .initial_state
        .as_ref()
        .and_then(|initial_state| initial_state.message_history.as_deref())
        .map(raw_messages_to_messages)
        .unwrap_or_default(),
    );

    for (index, step) in steps.iter().enumerate() {
        let tool_call_id = format!("tool_call_{}", index + 1);
        full_trajectory.push(Message::Assistant(AssistantMessage {
            role: "assistant",
            content: None,
            tool_calls: Some(vec![tool_call_from_function_call(&step.tool_call, &tool_call_id)]),
        }));
        full_trajectory.push(Message::Tool(ToolMessage {
            id: tool_call_id,
            role: "tool",
            content: Some(step.fc_output.clone()),
            requestor: step.tool_call.requestor,
            error: step.fc_output.contains("\"ok\":false"),
        }));
    }

    if let Some(final_answer) = final_answer {
        full_trajectory.push(Message::Assistant(AssistantMessage {
            role: "assistant",
            content: Some(final_answer.to_string()),
            tool_calls: None,
        }));
    }

    full_trajectory
}

fn raw_messages_to_messages(raw_messages: &[RawMessage]) -> Vec<Message> {
    raw_messages
        .iter()
        .filter_map(raw_message_to_message)
        .collect()
}

fn raw_message_to_message(raw_message: &RawMessage) -> Option<Message> {
    match raw_message.role.as_str() {
        "system" => Some(Message::System(SystemMessage {
            role: "system",
            content: raw_message.content.clone(),
        })),
        "user" => Some(Message::User(ParticipantMessageBase {
            role: "user",
            content: raw_message.content.clone(),
            tool_calls: raw_message
                .tool_calls
                .as_deref()
                .map(|tool_calls| tool_calls.iter().enumerate().map(|(index, tool_call)| tool_call_from_function_call(tool_call, &format!("raw_user_tool_call_{}", index + 1))).collect()),
        })),
        "assistant" => Some(Message::Assistant(ParticipantMessageBase {
            role: "assistant",
            content: raw_message.content.clone(),
            tool_calls: raw_message
                .tool_calls
                .as_deref()
                .map(|tool_calls| tool_calls.iter().enumerate().map(|(index, tool_call)| tool_call_from_function_call(tool_call, &format!("raw_assistant_tool_call_{}", index + 1))).collect()),
        })),
        "tool" => Some(Message::Tool(ToolMessage {
            id: "raw_tool_message".to_string(),
            role: "tool",
            content: raw_message.content.clone(),
            requestor: ToolRequestor::Assistant,
            error: false,
        })),
        _ => None,
    }
}

fn tool_call_from_function_call(function_call: &FunctionCall, id: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: function_call.name.clone(),
        arguments: function_call.arguments.clone(),
        requestor: function_call.requestor,
    }
}
