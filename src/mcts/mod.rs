use rand::distr::{Distribution, weighted::WeightedIndex};
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

use crate::{
    board::{Board, Winner},
    mcts::node::MCTSNode,
    model::EvalQueue,
};

mod node;

#[allow(type_alias_bounds)]
pub type History<B: Board> = (Vec<B>, Winner);

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

    fn winner(&self) -> Option<Winner> {
        self.board.winner()
    }

    async fn sample_move(&mut self, temperature: f32) {
        let legal_actions = self.board.legal_actions();

        let weights: Vec<f32> = self
            .root
            .lock()
            .await
            .num_visits
            .iter()
            .map(|n| (*n as f32).powf(1. / temperature))
            .collect();

        let action_idx = WeightedIndex::new(weights)
            .unwrap()
            .sample(&mut rand::rng());

        self.board.make_action(legal_actions[action_idx]);

        let child = self.root.lock().await.children.remove(action_idx).unwrap();
        self.root = child;
    }

    pub async fn create_history(
        eval_queue: EvalQueue,
        sims_per_move: usize,
        temperature: f32,
    ) -> History<B>
    where
        B: Default + Clone + Send + Sync + 'static,
        B::Action: Send,
    {
        let mut mcts = Self::new(eval_queue);

        let mut boards = Vec::new();

        let winner = loop {
            if let Some(winner) = mcts.winner() {
                break winner;
            }

            boards.push(mcts.board.clone());

            mcts.run_simulations(sims_per_move).await;
            mcts.sample_move(temperature).await;
        };

        (boards, winner)
    }
}
