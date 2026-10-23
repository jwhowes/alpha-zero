use burn::{Tensor, module::Module, tensor::Device};
use std::iter::zip;
use tokio::sync::{mpsc, oneshot};

use crate::{
    ViTConfig,
    board::{Board, BoardTensor, Player},
};

pub mod vit;

const MAX_BATCH_SIZE: usize = 4;

pub async fn evaluate<B: Board>(
    board: B,
    tx: &mpsc::Sender<(BoardTensor, oneshot::Sender<f32>)>,
) -> anyhow::Result<f32> {
    let player = board.player();

    let (v_tx, v_rx) = oneshot::channel();

    tx.send((board.to_tensor(), v_tx)).await?;

    let v = v_rx.await?;

    match player {
        Player::PlayerOne => Ok(v),

        Player::PlayerTwo => Ok(-v),
    }
}

pub async fn evaluation_thread(
    config: ViTConfig,
    device: &Device,
    mut rx: mpsc::Receiver<(BoardTensor, oneshot::Sender<f32>)>,
) -> anyhow::Result<()> {
    let model = config.init(device).no_grad();

    loop {
        let mut queue: Vec<(BoardTensor, oneshot::Sender<f32>)> =
            Vec::with_capacity(MAX_BATCH_SIZE);
        rx.recv_many(&mut queue, MAX_BATCH_SIZE).await;

        let (batch, tx): (Vec<BoardTensor>, Vec<oneshot::Sender<f32>>) = queue.into_iter().unzip();

        for (v, tx) in zip(
            model
                .forward(Tensor::cat(batch, 0).to_device(device))
                .try_into_vec_as::<f32>()?
                .into_iter(),
            tx.into_iter(),
        ) {
            let _ = tx.send(v);
        }
    }
}
