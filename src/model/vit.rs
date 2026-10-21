use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Device, activation as F},
};

#[derive(Module, Debug)]
pub struct SwiGLU {
    gate: Linear,
    hidden: Linear,
    out: Linear,
}

#[derive(Config, Debug)]
pub struct SwiGLUConfig {
    d_model: usize,
    #[config(default = "None")]
    d_hidden: Option<usize>,
}

impl SwiGLUConfig {
    pub fn init(&self, device: &Device) -> SwiGLU {
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
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.out
            .forward(F::silu(self.gate.forward(x.clone())) * self.hidden.forward(x))
    }
}

#[derive(Module, Debug)]
pub struct Attention {
    n_heads: usize,
    scale: f32,

    w_q: Linear,
    w_k: Linear,
    w_v: Linear,

    w_o: Linear,
}

#[derive(Config, Debug)]
pub struct AttentionConfig {
    n_heads: usize,
    d_model: usize,
}

impl AttentionConfig {
    pub fn init(&self, device: &Device) -> Attention {
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
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
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

        self.w_o
            .forward((F::softmax(attn, 3) / self.scale).matmul(v))
            .permute([0, 2, 3, 1])
            .flatten(2, 3)
    }
}
