use std::sync::mpsc::{Receiver, SyncSender};

use burn::{
    collective,
    collective::{CollectiveConfig, PeerId, ReduceOperation},
    prelude::Device,
    tensor::backend::AutodiffBackend,
};
use burn_optim::GradientsParams;

pub struct GradSyncer {
    sender: SyncSender<GradientsParams>,
    receiver: Receiver<Option<GradientsParams>>,
}

impl GradSyncer {
    pub fn start<B: AutodiffBackend>(
        config: CollectiveConfig,
        device: Device<B>,
        id: PeerId,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<GradientsParams>(1);
        let (result_sender, result_receiver) =
            std::sync::mpsc::sync_channel::<Option<GradientsParams>>(1);

        std::thread::spawn(move || {
            println!("[{id}] Register collective operation {config:?}");

            collective::register::<B::InnerBackend>(id, device, config).unwrap();

            while let Ok(grads) = receiver.recv() {
                let grads = grads
                    .all_reduce::<B::InnerBackend>(id, ReduceOperation::Mean)
                    .unwrap();

                result_sender.send(Some(grads)).unwrap();
            }

            collective::finish_collective::<B::InnerBackend>(id).unwrap();
        });

        Self {
            sender,
            receiver: result_receiver,
        }
    }

    pub fn sync(&self, grads: GradientsParams) -> Option<GradientsParams> {
        self.sender.send(grads).unwrap();
        self.receiver.recv().unwrap()
    }
}
