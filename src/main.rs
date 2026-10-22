use alpha_zero::ViTConfig;
use burn::tensor::{Device, Int, Tensor};

fn main() {
    let b = 8;
    let w = 4;
    let h = 4;
    let d_model = 64;
    let n_classes = 3;
    let n_blocks = 4;
    let n_heads = 4;

    let device = Device::wgpu(Default::default());

    let x = Tensor::<3, Int>::zeros([b, w, h], &device);

    let model = ViTConfig::new(n_classes, w, h, n_blocks, n_heads, d_model).init(&device);

    let y = model.forward(x);

    println!("{:?}", y.shape());
}
