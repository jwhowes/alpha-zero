use std::{f32::consts::SQRT_2, iter};

use rand::seq::IteratorRandom;

use crate::{
    board::{Board, Player, Winner},
    model::evaluate,
};

const GAMMA: f32 = 0.95;
const C_UCT: f32 = SQRT_2;

pub struct MCTSNode {
    winner: Option<Winner>,

    num_visits: Vec<usize>,
    total_action_value: Vec<f32>,

    children: Vec<Option<MCTSNode>>,
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

    pub fn run_simulation<B: Board>(&mut self, mut board: B, total_visits: f32) -> f32 {
        if let Some(winner) = board.winner() {
            return winner.value();
        }

        let player = board.player();
        let actions = board.legal_actions();

        let (action_idx, value) = if let Some((action_idx, _)) = self
            .children
            .iter()
            .enumerate()
            .find(|(_, c)| match c.as_ref() {
                None => false,

                Some(c) => match c.winner {
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
        } else {
            let leaves = self
                .children
                .iter_mut()
                .enumerate()
                .filter(|(_, c)| c.is_none());

            match leaves.choose(&mut rand::rng()) {
                None => {
                    // If no children are leaves, select the best
                    let action_idx = self
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

                    (
                        action_idx,
                        self.children[action_idx]
                            .as_mut()
                            .unwrap()
                            .run_simulation(board, self.num_visits[action_idx] as f32),
                    )
                }

                Some((action_idx, leaf)) => {
                    // If any children are leaves, select a random one

                    board.make_action(actions[action_idx]);

                    *leaf = Some(Self::new(&board));

                    (action_idx, evaluate(board))
                }
            }
        };

        self.num_visits[action_idx] += 1;
        self.total_action_value[action_idx] += match player {
            Player::PlayerOne => value,
            Player::PlayerTwo => -value,
        };

        GAMMA * value
    }
}
