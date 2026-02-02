pub mod host;
pub mod kernel;

#[derive(Clone, Debug)]
pub struct Wkv7ForwardOutput<T> {
    pub state: T,
    pub removal_state: T,
    pub output: T,
}

#[derive(Clone, Debug)]
pub struct Wkv7StatePassForwardOutput<T> {
    pub state: T,
    pub removal_state: T,
    pub output: T,
    pub final_state: T,
}

#[derive(Clone, Debug)]
pub struct Wkv7BackwardOutput<T> {
    pub weight_decay_grad: T,
    pub receptance_grad: T,
    pub key_grad: T,
    pub value_grad: T,
    pub removal_grad: T,
    pub replacement_grad: T,
}

#[derive(Clone, Debug)]
pub struct Wkv7StateBackwardOutput<T> {
    pub weight_decay_grad: T,
    pub receptance_grad: T,
    pub key_grad: T,
    pub value_grad: T,
    pub removal_grad: T,
    pub replacement_grad: T,
    pub initial_state_grad: T,
}
