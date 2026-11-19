# PIVOT research code release

The release implements the NeurIPS 2026 paper *Anchored Inertial Proposals with Objective Verification for Diffusion-Prior Inverse Problems*. The sampler, including curvature estimation, heavy-ball proposals, and final objective verification, is in [main.py](main.py).

The benchmark commands cover 32 dataset, prior, and task combinations. Each records reconstructions, per-draw quality metrics, sampling time, and peak allocated GPU memory. [Commands](configs_pivot_cli.txt) · [Checkpoints and data](docs/assets.md) · [Reproduction guide](docs/reproduction.md)

The release validation includes an exact 100-image FFHQ pixel super-resolution reproduction and targeted pixel- and latent-space comparisons. [Validation details](docs/validation.md) · [Paper on OpenReview](https://openreview.net/forum?id=8wurOw8o4D)
