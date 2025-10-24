use std::sync::mpsc::SyncSender;

use burn::{
    collective,
    collective::{CollectiveConfig, PeerId, ReduceOperation},
    prelude::Device,
    tensor::backend::AutodiffBackend,
};
use burn_optim::GradientsParams;

pub struct GradSyncer {
    sender: SyncSender<Message>,
}

pub struct Message {
    callback: SyncSender<Option<GradientsParams>>,
    grads: GradientsParams,
}

impl GradSyncer {
    pub fn start<B: AutodiffBackend>(
        config: CollectiveConfig,
        device: Device<B>,
        id: PeerId,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Message>(8);

        std::thread::spawn(move || {
            println!("[{id}] Register collective operation {config:?}");

            collective::register::<B::InnerBackend>(id, device, config).unwrap();

            let num_stages = 4;

            let mut buffers: Vec<GradientsParams> = Vec::new();

            while let Ok(msg) = receiver.recv() {
                let grads = msg
                    .grads
                    .all_reduce::<B::InnerBackend>(id, ReduceOperation::Mean)
                    .unwrap();

                buffers.push(grads);

                let result = if buffers.len() >= num_stages {
                    Some(buffers.remove(0))
                } else {
                    None
                };

                msg.callback.send(result).unwrap();
            }

            collective::finish_collective::<B::InnerBackend>(id).unwrap();
        });

        Self { sender }
    }

    pub fn sync(&self, grads: GradientsParams) -> Option<GradientsParams> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);

        let msg = Message {
            callback: sender,
            grads,
        };

        self.sender.send(msg).unwrap();

        receiver.recv().unwrap()
    }
}
