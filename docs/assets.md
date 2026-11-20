# Checkpoints and test images

Run the commands from the repository root with the environment in the README activated. For one experiment, download its dataset and prior. Nonlinear deblurring also uses the shared BKSE checkpoint.

## FFHQ pixel quick start

```bash
mkdir -p checkpoints dataset
gdown 'https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh' -O checkpoints/ffhq256.pt
gdown 'https://drive.google.com/uc?id=1i0oI8nt_b9XCHNPKM5KR92Y4t8ZVMDvR' -O dataset/test-ffhq.zip
unzip -q dataset/test-ffhq.zip -d dataset
python scripts/check_setup.py --checkpoint checkpoints/ffhq256.pt --data-root dataset/test-ffhq
```

The last command checks dependencies, CUDA availability, checkpoint presence, and image discovery. Continue with the [first reconstruction](../README.md#run-your-first-reconstruction).

## ImageNet pixel prior and test images

```bash
mkdir -p checkpoints dataset
gdown 'https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW' -O checkpoints/imagenet256.pt
gdown 'https://drive.google.com/uc?id=1ezXMhLt2UPaqNJnYNQAFM9ZLUW52ulz5' -O dataset/test-imagenet.zip
unzip -q dataset/test-imagenet.zip -d dataset
```

Both pixel and latent recipes use the same test images. The default roots are `dataset/test-ffhq/` and `dataset/test-imagenet/` in a fresh checkout. To use an existing image directory, pass `--data_root /path/to/images`. The loader finds PNG and JPEG files recursively, sorts the paths, converts images to RGB, resizes the shorter edge and center-crops to 256 × 256.

## Latent environment

Start from the pixel environment. Both CompVis priors use Lightning, OmegaConf and einops. The ImageNet conditioning module also imports Kornia and OpenAI CLIP. This command installs the extras for both:

```bash
python -m pip install pytorch-lightning==2.5.3 torchmetrics==1.8.1 omegaconf==2.3.0 einops==0.8.0 kornia==0.8.2 'git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'
```

Download the latent checkpoint for the dataset you want:

```bash
mkdir -p checkpoints
# FFHQ
curl -fL https://ommer-lab.com/files/latent-diffusion/ffhq.zip -o checkpoints/ffhq.zip
unzip -p checkpoints/ffhq.zip model.ckpt > checkpoints/ldm_ffhq256.pt

# ImageNet
curl -fL https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt -o checkpoints/ldm_imagenet256.pt
```

Choose `--model ffhq256ldm` or `--model imagenet256ldm`. The ImageNet latent prior receives the null class label 1000. The runner decodes continuously for measurement gradients and for the final image. To use an existing prior checkpoint, pass `--ldm_diffusion_path` for latent diffusion or `--ddpm_model_path` for pixel diffusion. [CompVis latent diffusion](https://github.com/CompVis/latent-diffusion)

## Nonlinear blur

```bash
mkdir -p checkpoints
gdown 'https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy' -O checkpoints/GOPRO_wVAE.pth
```

The `nonlinear_blur` operator loads these weights for both priors and both datasets. The data and pretrained models are distributed by their upstream providers. [BKSE](https://github.com/VinAIResearch/blur-kernel-space-exploring)
