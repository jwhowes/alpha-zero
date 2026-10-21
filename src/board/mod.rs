use burn::Tensor;

#[derive(Clone, Copy, PartialEq)]
pub enum Player {
    PlayerOne,
    PlayerTwo,
}

#[derive(Clone, Copy)]
pub enum Winner {
    Tie,
    Player(Player),
}

impl Winner {
    pub fn value(&self) -> f32 {
        match self {
            Self::Tie => 0.,
            Self::Player(Player::PlayerOne) => 1.,
            Self::Player(Player::PlayerTwo) => -1.,
        }
    }
}

pub trait Board {
    type Action: Copy;

    fn player(&self) -> Player;
    fn winner(&self) -> Option<Winner>;

    fn legal_actions(&self) -> Vec<Self::Action>;
    fn make_action(&mut self, action: Self::Action);

    fn to_tensor(self) -> Tensor<4>;
}
