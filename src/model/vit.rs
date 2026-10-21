use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{Device, Distribution, Int, activation as F, s},
};

#[derive(Module, Debug)]
struct SwiGLU {
    gate: Linear,
    hidden: Linear,
    out: Linear,
}

#[derive(Config, Debug)]
struct SwiGLUConfig {
    d_model: usize,
    #[config(default = "None")]
    d_hidden: Option<usize>,
}

impl SwiGLUConfig {
    fn init(&self, device: &Device) -> SwiGLU {
        let d_hidden = self.d_hidden.unwrap_or(4 * self.d_model);

        let linear_config = LinearConfig::new(self.d_model, d_hidden).with_bias(false);

        SwiGLU {
            gate: linear_config.init(device),
            hidden: linear_config.init(device),

            out: LinearConfig::new(d_hidden, self.d_model).init(device),
        }
    }
}

impl SwiGLU {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.out
            .forward(F::silu(self.gate.forward(x.clone())) * self.hidden.forward(x))
    }
}

#[derive(Module, Debug)]
struct Attention {
    n_heads: usize,
    scale: f32,

    w_q: Linear,
    w_k: Linear,
    w_v: Linear,

    w_o: Linear,
}

#[derive(Config, Debug)]
struct AttentionConfig {
    n_heads: usize,
    d_model: usize,
}

impl AttentionConfig {
    fn init(&self, device: &Device) -> Attention {
        let linear_config = LinearConfig::new(self.d_model, self.d_model).with_bias(false);

        Attention {
            n_heads: self.n_heads,
            scale: ((self.d_model as f32) / (self.n_heads as f32)).sqrt(),

            w_q: linear_config.init(device),
            w_k: linear_config.init(device),
            w_v: linear_config.init(device),

            w_o: linear_config.init(device),
        }
    }
}

impl Attention {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        // x: B x L x d
        let q = self
            .w_q
            .forward(x.clone())
            .unfold(2, self.n_heads, self.n_heads)
            .permute([0, 3, 1, 2]);
        let k = self
            .w_k
            .forward(x.clone())
            .unfold(2, self.n_heads, self.n_heads)
            .permute([0, 3, 2, 1]);
        let v = self
            .w_v
            .forward(x)
            .unfold(2, self.n_heads, self.n_heads)
            .permute([0, 3, 1, 2]);

        let attn = q.matmul(k);

        self.w_o.forward(
            (F::softmax(attn, 3) / self.scale)
                .matmul(v)
                .permute([0, 2, 3, 1])
                .flatten(2, 3),
        )
    }
}

#[derive(Module, Debug)]
struct ViTBlock {
    attn: Attention,
    attn_norm: RmsNorm,

    ffn: SwiGLU,
    ffn_norm: RmsNorm,
}

#[derive(Config, Debug)]
struct ViTBlockConfig {
    d_model: usize,
    n_heads: usize,

    #[config(default = "None")]
    d_hidden: Option<usize>,

    #[config(default = "1e-6")]
    norm_eps: f32,
}

impl ViTBlockConfig {
    fn init(&self, device: &Device) -> ViTBlock {
        let norm_config = RmsNormConfig::new(self.d_model).with_epsilon(self.norm_eps as f64);

        ViTBlock {
            attn: AttentionConfig::new(self.n_heads, self.d_model).init(device),
            attn_norm: norm_config.init(device),

            ffn: SwiGLUConfig::new(self.d_model)
                .with_d_hidden(self.d_hidden)
                .init(device),
            ffn_norm: norm_config.init(device),
        }
    }
}

impl ViTBlock {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.attn_norm.forward(x));

        x.clone() + self.ffn.forward(self.ffn_norm.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct ViT {
    cls_emb: Param<Tensor<3>>,
    pos_emb: Param<Tensor<3>>,

    emb: Embedding,

    blocks: Vec<ViTBlock>,

    head: Linear,
}

#[derive(Config, Debug)]
pub struct ViTConfig {
    n_classes: usize,

    board_width: usize,
    board_height: usize,

    n_blocks: usize,
    n_heads: usize,
    d_model: usize,

    n_actions: usize,
}

impl ViTConfig {
    pub fn init(&self, device: &Device) -> ViT {
        let l = self.board_width * self.board_height;

        let dist = Distribution::Normal(0., 0.1);

        let vit_block_config = ViTBlockConfig::new(self.d_model, self.n_heads);

        ViT {
            cls_emb: Param::from_tensor(Tensor::random([1, 1, self.d_model], dist, device)),
            pos_emb: Param::from_tensor(Tensor::random([1, l, self.d_model], dist, device)),

            emb: EmbeddingConfig::new(self.n_classes, self.d_model).init(device),

            blocks: (0..self.n_blocks)
                .map(|_| vit_block_config.init(device))
                .collect(),

            head: LinearConfig::new(self.d_model, self.n_actions)
                .with_bias(false)
                .init(device),
        }
    }
}

impl ViT {
    pub fn forward(&self, board: Tensor<3, Int>) -> Tensor<2> {
        let b = board.shape()[0];

        let x = self.emb.forward(board.flatten::<2>(1, 2));

        let mut x = Tensor::cat(
            vec![
                self.cls_emb.val().expand([b as isize, -1, -1]),
                x + self.pos_emb.val(),
            ],
            1,
        );

        for block in &self.blocks {
            x = block.forward(x);
        }

        self.head.forward(x.slice(s![.., 0]).squeeze())
    }
}
