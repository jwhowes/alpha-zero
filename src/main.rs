use burn::tensor::{Tensor, activation as F};

fn main() {
    let b = 8;
    let l = 12;
    let d = 64;
    let n_heads = 4;

    let scale = ((d as f32) / (n_heads as f32)).sqrt();

    let device = Default::default();

    let q: Tensor<4> = Tensor::<3>::zeros([b, l, d], &device)
        .unfold(2, n_heads, n_heads)
        .permute([0, 3, 1, 2]);
    let k: Tensor<4> = Tensor::<3>::zeros([b, l, d], &device)
        .unfold(2, n_heads, n_heads)
        .permute([0, 3, 2, 1]);
    let v: Tensor<4> = Tensor::<3>::zeros([b, l, d], &device)
        .unfold(2, n_heads, n_heads)
        .permute([0, 3, 1, 2]);

    let attn = q.matmul(k);

    println!("{:?}", attn.shape());

    let x = (F::softmax(attn, 3) / scale).matmul(v);

    println!("{:?}", [b, n_heads, l, d / n_heads]);
    println!("{:?}", x.shape());

    let y: Tensor<3> = x.permute([0, 2, 3, 1]).flatten(2, 3);

    println!("{:?}", y.shape());
}
