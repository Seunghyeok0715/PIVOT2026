# Release validation

The release entry point was compared with the PIVOT reference implementation using the same checkpoints, measurements and random seeds. The GPU comparisons below used the existing Linux environment with Python 3.10, PyTorch 2.6.0+cu118, torchvision 0.21.0 and RTX 6000 Ada GPUs. The installation example uses the versions pinned in the repository's pixel requirements.

| Comparison | Result |
| --- | --- |
| FFHQ pixel 4× SR, 100 images, K=40, J=15, seed 42 | All 100 PNGs matched byte for byte. All per-image PSNR, SSIM and LPIPS values matched. Means were 29.666 dB, 0.847 and 0.249 |
| FFHQ and ImageNet pixel tasks, 121 images including the SR run | 115 PNGs matched byte for byte. In six Gaussian- or motion-blur images the largest RGB channel difference was one 8-bit level. Phase retrieval was checked on the first draw |
| FFHQ latent 4× SR, first two images, K=50, J=24 | Both PNGs, PSNR and SSIM matched. The largest LPIPS difference from the stored ten-image evaluation was 0.000002 |
| ImageNet class-conditional latent HDR, one image, K=4, J=3 | The full floating-point reconstruction, all three metrics and final GPU random-number state matched across the two implementations |

For Gaussian and motion blur, repeating the reference code on the same GPU also changed some channels by one 8-bit level. With inputs and weights held fixed, the CUDA backward pass through the operators' reflection padding produced small differences at the image boundary.

The CPU suite covers image loading and slicing, the benchmark commands, and the verifier's decision to accept a proposal or retain the anchor. Run it from the repository root:

```bash
python -m unittest discover -s tests -v
```

The paper's 100-image results for all tasks are listed separately. [Reported benchmark results](results.md)
