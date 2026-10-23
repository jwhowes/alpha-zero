use std::{f32::consts::SQRT_2, iter, sync::Arc};

use tokio::sync::Mutex;

use crate::{
    board::{Board, Player, Winner},
    model::{EvalQueue, evaluate},
};

const GAMMA: f32 = 0.95;
const C_UCT: f32 = SQRT_2;
const VIRTUAL_LOSS: i32 = 3;

pub struct MCTSNode {
    winner: Option<Winner>,

    num_visits: Vec<i32>,
    total_action_value: Vec<f32>,

    children: Vec<Option<Arc<Mutex<MCTSNode>>>>,
}

impl MCTSNode {
    pub fn new<B: Board>(board: &B) -> Self {
        let n = board.legal_actions().len();

        Self {
            winner: board.winner(),

            num_visits: Vec::from_iter(iter::repeat_n(0, n)),
            total_action_value: Vec::from_iter(iter::repeat_n(0., n)),

            children: Vec::from_iter((0..n).map(|_| None)),
        }
    }

    fn value(&self, total_visits: f32) -> Vec<f32> {
        (0..self.children.len())
            .map(|i| {
                (self.total_action_value[i] / (self.num_visits[i] as f32))
                    + C_UCT * (total_visits / (self.num_visits[i] as f32)).sqrt()
            })
            .collect()
    }

    pub async fn run_simulation<B: Board>(
        node: Arc<Mutex<Self>>,
        mut board: B,
        total_visits: f32,
        eval_queue: &EvalQueue,
    ) -> f32 {
        if let Some(winner) = board.winner() {
            return winner.value();
        }

        let player = board.player();
        let actions = board.legal_actions();

        let (action_idx, value) = if let Some((action_idx, _)) = node
            .lock()
            .await
            .children
            .iter()
            .enumerate()
            .find(|(_, c)| match c.as_ref() {
                None => false,

                Some(c) => match c.blocking_lock().winner {
                    None | Some(Winner::Tie) => false,

                    Some(Winner::Player(p)) => p == player,
                },
            }) {
            // If a child is a winner for us, make that move

            (
                action_idx,
                match player {
                    Player::PlayerOne => 1.,
                    Player::PlayerTwo => -1.,
                },
            )
        } else if let Some((action_idx, leaf)) = node
            .lock()
            .await
            .children
            .iter_mut()
            .enumerate()
            .filter(|(_, c)| c.is_none())
            .next()
        {
            // The leaf is locked during evaluation
            board.make_action(actions[action_idx]);

            *leaf = Some(Arc::new(Mutex::new(Self::new(&board))));

            (action_idx, evaluate(board, eval_queue).await.unwrap())
        } else {
            // If no children are leaves, select the best
            let action_idx = node
                .lock()
                .await
                .value(total_visits)
                .into_iter()
                .enumerate()
                .max_by(|(_, v1), (_, v2)| {
                    if v1 < v2 {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
                .unwrap()
                .0;

            board.make_action(actions[action_idx]);

            let (child, total_visits) = {
                let mut lock = node.lock().await;

                let total_visits = lock.num_visits[action_idx];
                lock.num_visits[action_idx] += VIRTUAL_LOSS;

                (
                    Arc::clone(lock.children[action_idx].as_ref().unwrap()),
                    total_visits as f32,
                )
            };

            let v = Self::run_simulation(child, board, total_visits, eval_queue).await;

            node.lock().await.num_visits[action_idx] -= VIRTUAL_LOSS;

            (action_idx, v)
        };

        {
            let mut lock = node.lock().await;

            lock.num_visits[action_idx] += 1;
            lock.total_action_value[action_idx] += match player {
                Player::PlayerOne => value,
                Player::PlayerTwo => -value,
            };
        }

        GAMMA * value
    }
}
