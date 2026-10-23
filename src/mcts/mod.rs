use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

use crate::{board::Board, mcts::node::MCTSNode, model::EvalQueue};

mod node;

pub struct MCTS<const N_THREADS: usize, B: Board> {
    total_visits: f32,
    eval_queue: EvalQueue,

    board: B,

    root: Arc<Mutex<MCTSNode>>,
}

impl<const N_THREADS: usize, B: Board> MCTS<N_THREADS, B> {
    pub fn new(eval_queue: EvalQueue) -> Self
    where
        B: Default,
    {
        let board = B::default();

        Self {
            total_visits: 1.,
            eval_queue,

            root: Arc::new(Mutex::new(MCTSNode::new(&board))),
            board,
        }
    }

    async fn simulation_thread(
        root: Arc<Mutex<MCTSNode>>,
        total_visits: f32,
        board: B,
        eval_queue: EvalQueue,
        num_simulations: usize,
        count: Arc<Mutex<usize>>,
    ) where
        B: Clone,
    {
        loop {
            if *count.lock().await >= num_simulations {
                break;
            } else {
                *count.lock().await += 1;
            }

            MCTSNode::run_simulation(root.clone(), board.clone(), total_visits, &eval_queue).await;
        }
    }

    async fn run_simulations(&self, num_simulations: usize)
    where
        B: Clone + Send + Sync + 'static,
        B::Action: Send,
    {
        let count = Arc::new(Mutex::new(0));

        let handles = (0..N_THREADS)
            .map(move |_| {
                let total_visits = self.total_visits;
                let root = Arc::clone(&self.root);
                let count = Arc::clone(&count);
                let board = self.board.clone();
                let root = Arc::clone(&root);
                let eval_queue = mpsc::Sender::clone(&self.eval_queue);

                tokio::spawn(async move {
                    Self::simulation_thread(
                        root,
                        total_visits,
                        board,
                        eval_queue,
                        num_simulations,
                        count,
                    )
                    .await
                })
            })
            .collect::<Vec<_>>();

        for h in handles {
            let _ = h.await;
        }
    }
}
