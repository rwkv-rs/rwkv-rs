#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Completed,
    Failed,
}

impl TaskStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "Running" => Ok(Self::Running),
            "Completed" => Ok(Self::Completed),
            "Failed" => Ok(Self::Failed),
            other => Err(format!("unknown task status `{other}`")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionStatus {
    Running,
    Completed,
    Failed,
}

impl CompletionStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
        }
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "Running" => Ok(Self::Running),
            "Completed" => Ok(Self::Completed),
            "Failed" => Ok(Self::Failed),
            other => Err(format!("unknown completion status `{other}`")),
        }
    }
}
