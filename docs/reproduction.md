# Reproducing the natural-image benchmarks

Run each recipe from the repository root. One command evaluates one dataset, prior and task. To run all 32 combinations sequentially on GPU 0 after installing all assets, use the following command. [Setup](assets.md) · [Individual commands](../configs_pivot_cli.txt)

```bash
bash -e configs_pivot_cli.txt
```

## Inputs and random draws

Each command evaluates sorted image indices 0 through 99 with batch size 1 and seed 42. The measurement operator adds Gaussian noise with standard deviation 0.05. The solver uses `--sigma_y 0.05` to scale its Gaussian likelihood. Changing this flag affects the reconstruction objective and leaves the operator's noise setting in its preset.

The runner seeds NumPy and PyTorch once per process. For each first-run batch it generates the measurement and then draws the initial state. Later runs reuse that measurement and draw new initial states. Each nonterminal reverse step draws fresh noise. Keep the original batch size, image range and run count together in one invocation to replay a benchmark sequence. CUDA reflection-padding gradients used for Gaussian and motion blur can vary slightly between repeats. [Release validation](validation.md)

## Files and metrics

The default directory is `<save_dir>/<task>/`. Pass a distinct `--name` or `--save_dir` to keep multiple experiments. For example, the FFHQ pixel SR benchmark writes to `results/ffhq_pixel/down_sampling/`.

| File | Contents |
| --- | --- |
| `config.yaml` | Resolved data, operator, model, scheduler, and CLI arguments |
| `samples/00000_run0000.png` | One reconstruction per image and draw. Filenames use the index within the selected slice, starting at zero |
| `grid_results.png` | Rows of ground truth, resized measurements, and reconstructions for each draw |
| `metrics.json` | Per-image, per-draw PSNR, SSIM, LPIPS, measurement residuals, and sampling diagnostics |
| `eval.md` | Per-image scores and their mean and standard deviation across images |
| `sampling_diagnostics.json` | Run and image IDs, sampling seconds, peak allocated GPU memory, number of levels, $J$, and likelihood scale |

The runner scores the floating-point reconstructions before PNG quantization. PSNR and SSIM use PIQ, and LPIPS uses the VGG backbone from the `lpips` package. Phase retrieval uses four draws per image with the same measurement. [Table 1 results](results.md)

### Runtime and memory

Sampling time sums the synchronized reverse levels, including the denoiser, proposal, final verifier, and reinsertion. Measurement generation, model initialization, final latent decoding, metric evaluation, and image writing are outside this timer. Mean seconds per draw equal the sum of `sampling_time_sec` divided by the total number of reconstructions. For four-draw phase retrieval, multiply this mean by four to obtain mean sampling seconds per input image.

On CUDA, `peak_memory_gb` gives peak **allocated GiB** for each sampling batch. It includes resident models and the final latent decode. The benchmark uses batch size 1.
