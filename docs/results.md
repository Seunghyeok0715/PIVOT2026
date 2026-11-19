# Natural-image reconstruction results

These are the author-reported PIVOT results from Table 1. Each task uses 100 FFHQ images and 100 ImageNet images at 256 × 256. Runtime is the mean sampling time in seconds per reconstruction on one NVIDIA RTX 6000 Ada GPU at batch size 1. For phase retrieval it is the time for one of four trials. PSNR is in dB. Higher PSNR and SSIM and lower LPIPS indicate better reconstruction quality. [PIVOT on OpenReview](https://openreview.net/forum?id=8wurOw8o4D)

## Pixel-space PIVOT

| Task | FFHQ PSNR ↑ | FFHQ SSIM ↑ | FFHQ LPIPS ↓ | FFHQ time (s) | ImageNet PSNR ↑ | ImageNet SSIM ↑ | ImageNet LPIPS ↓ | ImageNet time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4× super-resolution | 29.666 | 0.847 | 0.249 | 1.305 | 26.618 | 0.731 | 0.325 | 2.398 |
| Box inpainting | 25.131 | 0.860 | 0.156 | 2.403 | 21.705 | 0.803 | 0.230 | 5.159 |
| Random inpainting | 32.641 | 0.914 | 0.144 | 2.644 | 29.975 | 0.858 | 0.181 | 5.738 |
| Gaussian deblurring | 29.630 | 0.842 | 0.239 | 3.675 | 26.516 | 0.715 | 0.338 | 4.540 |
| Motion deblurring | 32.413 | 0.894 | 0.173 | 1.627 | 29.849 | 0.834 | 0.212 | 3.282 |
| Phase retrieval | 31.343 | 0.880 | 0.205 | 24.051 | 20.958 | 0.547 | 0.435 | 28.194 |
| Nonlinear deblurring | 29.939 | 0.848 | 0.221 | 151.812 | 28.562 | 0.795 | 0.259 | 157.840 |
| High dynamic range | 26.013 | 0.859 | 0.186 | 1.537 | 25.379 | 0.848 | 0.190 | 2.548 |

## Latent-space PIVOT

| Task | FFHQ PSNR ↑ | FFHQ SSIM ↑ | FFHQ LPIPS ↓ | FFHQ time (s) | ImageNet PSNR ↑ | ImageNet SSIM ↑ | ImageNet LPIPS ↓ | ImageNet time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4× super-resolution | 29.033 | 0.824 | 0.292 | 39.816 | 25.998 | 0.695 | 0.402 | 39.112 |
| Box inpainting | 24.584 | 0.838 | 0.243 | 80.738 | 21.229 | 0.751 | 0.341 | 78.573 |
| Random inpainting | 31.644 | 0.890 | 0.210 | 32.613 | 28.327 | 0.801 | 0.282 | 33.775 |
| Gaussian deblurring | 28.047 | 0.796 | 0.325 | 50.341 | 24.965 | 0.644 | 0.446 | 50.784 |
| Motion deblurring | 29.936 | 0.838 | 0.278 | 50.678 | 26.791 | 0.717 | 0.378 | 54.401 |
| Phase retrieval | 27.925 | 0.788 | 0.304 | 79.183 | 18.166 | 0.444 | 0.589 | 82.102 |
| Nonlinear deblurring | 29.131 | 0.824 | 0.287 | 200.075 | 26.279 | 0.710 | 0.379 | 213.401 |
| High dynamic range | 25.050 | 0.826 | 0.260 | 63.385 | 25.547 | 0.779 | 0.292 | 65.486 |
