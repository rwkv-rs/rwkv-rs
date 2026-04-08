use crate::cores::datasets::function_calling::FunctionCall;

use super::super::data_model::message::Message;

pub struct EvaluatorBase;

impl EvaluatorBase {
    pub fn extract_predicted_tool_calls(full_trajectory: &[Message]) -> Vec<FunctionCall> {
        let mut predicted_tool_calls = Vec::new();
        for message in full_trajectory {
            let Some(message) = message.participant_message() else {
                continue;
            };
            if !message.is_tool_call() {
                continue;
            }
            for tool_call in message.tool_calls.as_deref().unwrap_or(&[]) {
                predicted_tool_calls.push(FunctionCall {
                    requestor: tool_call.requestor,
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                });
            }
        }
        predicted_tool_calls
    }
}
