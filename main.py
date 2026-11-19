"""PIVOT for pixel- and latent-space diffusion inverse problems."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from torch.func import jvp as _func_jvp, vjp as _func_vjp  # PyTorch >= 2.0
except Exception:  # noqa: BLE001
    try:
        from functorch import jvp as _func_jvp, vjp as _func_vjp  # type: ignore
    except Exception:  # noqa: BLE001
        _func_jvp = None  # type: ignore
        _func_vjp = None  # type: ignore

from torch.utils.data import DataLoader
from torchvision.utils import save_image
import yaml
import wandb
import setproctitle

import tqdm
import torch.nn as nn

from data import get_dataset
from forward_operators import get_operator, LatentWrapper
from model import get_model
from misc import (
    # Presets extracted from upstream YAMLs (configs/*/*.yaml)
    DATA_PRESETS,
    MODEL_PRESETS,
    SAMPLER_PRESETS,
    TASK_PRESETS,

    # Scheduler registry + evaluation + utilities
    Trajectory,
    get_diffusion_scheduler,
    Evaluator,
    get_eval_fn,
    get_eval_fn_cmp,
    calculate_fid,
    resize,
    safe_dir,
    tensor_to_pils,
    save_mp4_video,
)


class _GatedClip(torch.autograd.Function):
    """Keep the clipped forward value. Pass significant inward gradients at saturation."""

    @staticmethod
    def forward(ctx, value, threshold):
        ctx.save_for_backward(value)
        ctx.threshold = threshold
        return value.clamp(-1, 1)

    @staticmethod
    def backward(ctx, grad):
        (value,) = ctx.saved_tensors
        inward = ((value > 1) & (grad > 0)) | ((value < -1) & (grad < 0))
        keep = (value.abs() <= 1) | (inward & (grad.abs() > ctx.threshold))
        return grad * keep, None


def _observation(operator, image, sigma_y, clip_grad):
    """Use the operator's exact forward value for both proposals and verification."""
    if clip_grad == "exact":
        return operator(image)
    name = getattr(operator, "name", None)
    if name == "high_dynamic_range":
        value = image * operator.scale
    elif name == "nonlinear_blur":
        data = (image + 1.0) / 2.0
        kernel = operator.random_kernel.repeat(data.shape[0], 1, 1, 1)
        value = operator.blur_model.adaptKernel(data, kernel=kernel) * 2.0 - 1.0
    else:
        return operator(image)
    sigma_n = float(operator.sigma)
    scale = float(np.sqrt(2.0 * np.log(value[0].numel())))
    threshold = scale / sigma_n if sigma_y == sigma_n else scale * sigma_n / (sigma_y * sigma_y)
    return _GatedClip.apply(value, threshold)


def _dot(a, b):
    """One inner product per image, in the input dtype."""
    return (a.flatten(1) * b.flatten(1)).sum(dim=1)


def _expand(values, state):
    return values.to(state.dtype).view(-1, *([1] * (state.ndim - 1)))


class Pivot:
    def __init__(self, scheduler_config, *, latent, J, sigma_y, control="none", clip_grad="gate"):
        self.scheduler = get_diffusion_scheduler(**scheduler_config)
        self.latent = latent
        self.J = J
        self.sigma_y = sigma_y
        self.control = control
        self.clip_grad = clip_grad

    def get_start(self, batch_size, model):
        device = next(model.parameters()).device
        return torch.randn(batch_size, *model.get_in_shape(), device=device) * self.scheduler.get_prior_sigma()

    def _denoise(self, state, sigma):
        sigma = state.new_tensor(max(sigma, 1e-3))
        if getattr(self.model, "is_conditional", False):
            labels = torch.full((state.shape[0],), 1000, device=state.device, dtype=torch.long)
            return self.model.model(state, sigma, class_labels=self.model.net.get_condition(labels))
        return self.model.tweedie(state, sigma)

    def _decode(self, state):
        if self.latent:
            return self.model.net.model.differentiable_decode_first_stage(state, force_not_quantize=True)
        return state

    def _loss(self, z):
        pred = _observation(self.operator, self._decode(z), self.sigma_y, self.clip_grad)
        loss = (0.5 * ((pred - self.y) / self.sigma_y).square()).flatten(1).sum(dim=1)
        return torch.where(torch.isfinite(loss), loss, torch.full_like(loss, float("inf")))

    def _loss_grad(self, z):
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            loss = self._loss(z)
            grad = torch.autograd.grad(loss.sum(), z)[0]
        return loss.detach(), torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    def _level(self, anchor, beta, curvature):
        """Take J-1 inertial steps, then verify the proposal against the frozen anchor."""
        beta = torch.full((anchor.shape[0],), beta, dtype=torch.float64, device=anchor.device)
        anchor_loss, data_grad = self._loss_grad(anchor)
        if curvature is None:
            # One relative probe per trajectory initializes the carried curvature.
            grad_norm = _dot(data_grad, data_grad).sqrt().clamp(min=1e-30)
            anchor_norm = _dot(anchor, anchor).sqrt().clamp(min=1e-6)
            probe = anchor - _expand(1e-3 * anchor_norm / grad_norm, anchor) * data_grad
            _, probe_grad = self._loss_grad(probe)
            diff = probe_grad - data_grad
            delta = probe - anchor
            curvature = (_dot(diff, diff).sqrt() / _dot(delta, delta).sqrt().clamp(min=1e-30)).double().clamp(min=1e-12)

        working = curvature
        max_secant = torch.zeros_like(curvature)
        z = prev = anchor
        grad = data_grad
        phi = anchor_loss.double()
        relative_eps = torch.finfo(z.dtype).eps ** 0.5
        mask_shape = (z.shape[0],) + (1,) * (z.ndim - 1)
        for j in range(self.J - 1):
            upper = working + beta
            if (j == 0 and self.control != "polyak_first") or self.control == "no_momentum":
                candidate = z - _expand(1.0 / upper, z) * grad
            else:
                hi, lo = upper.sqrt(), beta.sqrt()
                alpha = 4.0 / (hi + lo).square()
                gamma = ((hi - lo) / (hi + lo)).square()
                candidate = z - _expand(alpha, z) * grad + _expand(gamma, z) * (z - prev)

            loss, next_grad = self._loss_grad(candidate)
            delta = candidate - anchor
            next_phi = loss.double() + 0.5 * beta * _dot(delta, delta).double()
            step = candidate - z
            step_norm = _dot(step, step).sqrt()
            diff = next_grad - data_grad
            secant = (_dot(diff, diff).sqrt() / step_norm.clamp(min=1e-30)).double()
            valid = torch.isfinite(secant) & (step_norm > relative_eps * _dot(z, z).sqrt())
            working = torch.where(valid, torch.maximum(working, secant), working)
            max_secant = torch.where(valid, torch.maximum(max_secant, secant), max_secant)
            restart = ~(next_phi <= phi) | ~torch.isfinite(next_phi)
            base = torch.maximum(curvature, max_secant) if self.control == "restart_measured" else working
            working = torch.where(restart, 2.0 * base, working)
            prev = torch.where(restart.view(mask_shape), candidate, z)
            z, data_grad, phi = candidate, next_grad, next_phi
            grad = data_grad + _expand(beta, z) * (z - anchor)

        carry = curvature if self.control == "max_carry" else 0.5 * curvature
        curvature = torch.maximum(max_secant, carry)
        if self.control == "no_verify":
            return z, curvature
        # Re-form the endpoint before scoring, as in the benchmark's verifier.
        delta = z - anchor
        candidate = anchor + delta
        verified_phi = self._loss(candidate).double() + 0.5 * beta * _dot(delta, delta).double()
        accepted = verified_phi < anchor_loss.double()
        return torch.where(accepted.view(mask_shape), candidate, anchor), curvature

    @torch.no_grad()
    def sample(self, model, x_start, operator, y):
        self.model, self.operator, self.y = model, operator, y
        state = x_start.detach()
        device = state.device
        sigmas = self.scheduler.sigma_steps.to(state).tolist()
        if state.is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = 0.0
        curvature = None
        for sigma, sigma_next in zip(sigmas, sigmas[1:]):
            if state.is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            anchor = self._denoise(state, sigma)
            proposal, curvature = self._level(anchor, 1.0 / max(sigma, 1e-3) ** 2, curvature)
            noise = torch.randn_like(state) if sigma_next > 0 else torch.zeros_like(state)
            state = proposal + sigma_next * noise
            if state.is_cuda:
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
        out = self._decode(state)
        self.diagnostics = {
            "sampling_time_sec": elapsed, "levels": len(sigmas) - 1, "J": self.J, "sigma_y": self.sigma_y,
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if state.is_cuda else None,
        }
        return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", choices=sorted(DATA_PRESETS), default="test-ffhq")
    p.add_argument("--model", choices=("ffhq256ddpm", "imagenet256ddpm", "ffhq256ldm", "imagenet256ldm"),
                   default="ffhq256ddpm")
    p.add_argument("--task", choices=sorted(TASK_PRESETS), default="down_sampling")
    p.add_argument("--K", type=int, required=True, help="number of reverse diffusion levels")
    p.add_argument("--J", type=int, required=True, help="data gradient evaluations per level, excluding setup")
    p.add_argument("--sigma_y", type=float, default=0.05, help="Gaussian likelihood standard deviation")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data_root")
    p.add_argument("--data_start_id", type=int)
    p.add_argument("--data_end_id", type=int)
    p.add_argument("--ddpm_model_path")
    p.add_argument("--ldm_diffusion_path")
    p.add_argument("--save_dir", default="./results")
    p.add_argument("--name", help="output subfolder (defaults to the task name)")
    p.add_argument("--eval_fn_list", default="psnr,ssim,lpips")
    p.add_argument("--control", default="none", help="paper ablation (default: full PIVOT)",
                   choices=("none", "no_momentum", "polyak_first", "max_carry", "restart_measured", "no_verify"))
    p.add_argument("--clip_grad", choices=("gate", "exact"), default="gate")
    args = p.parse_args()
    if min(args.K, args.J, args.batch_size, args.num_runs) < 1 or not (0 < args.sigma_y < float("inf")):
        p.error("K, J, batch_size, num_runs and sigma_y must be positive and finite")
    args.name = args.name or args.task
    return args


def build_configs(args):
    data = dict(DATA_PRESETS[args.data])
    for key, value in (("root", args.data_root), ("start_id", args.data_start_id), ("end_id", args.data_end_id)):
        if value is not None:
            data[key] = value
    model = dict(MODEL_PRESETS[args.model])
    latent = model["name"] == "ldm"
    if latent and args.ldm_diffusion_path:
        model["diffusion_path"] = args.ldm_diffusion_path
    if not latent and args.ddpm_model_path:
        model["model_config"] = dict(model["model_config"], model_path=args.ddpm_model_path)
    operator = dict(TASK_PRESETS[args.task]["ldm" if latent else "pixel"]["operator"])
    if "resolution" in operator:
        operator["resolution"] = data["resolution"]
    scheduler = {"name": "edm", "num_steps": args.K, "sigma_max": 10.0 if latent else 80.0,
                 "sigma_min": 0.02, "timestep": "poly-7"}
    return data, model, operator, scheduler


def _load_prior(config, device):
    def direct(func, inputs, params, flag):
        return func(*inputs)

    import model.ddpm.nn as ddpm_nn
    import model.ddpm.unet as ddpm_unet
    ddpm_nn.checkpoint = ddpm_unet.checkpoint = direct
    if config["name"] == "ldm":
        import model.ldm.modules.diffusionmodules.util as ldm_util
        ldm_util.checkpoint = direct
        # Resolve the checkpoint's model class before opening the weights. CompVis
        # imports rank_zero_only from a path that moved in Lightning 2.x.
        module_name = config["ldm_config"]["model"]["target"].rsplit(".", 1)[0]
        try:
            __import__(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != "pytorch_lightning.utilities.distributed":
                raise
            os.sys.modules[exc.name] = os.sys.modules["pytorch_lightning.utilities.rank_zero"]
    return get_model(**config, device=device)


@torch.no_grad()
def _fidelity(operator, y, samples):
    runs, count = samples.shape[:2]
    target = y.unsqueeze(0).expand(runs, *y.shape).flatten(0, 1)
    residual = operator(samples.flatten(0, 1)) - target
    energy = residual.abs().square().flatten(1).sum(dim=1)
    rms = (energy / residual[0].numel()).sqrt()
    relative = (energy / target.abs().square().flatten(1).sum(dim=1).clamp(min=1e-24)).sqrt()
    return {"definition": {"rms": "||A(x)-y||_2/sqrt(number of measurement elements)",
                           "relative_l2": "||A(x)-y||_2/||y||_2"},
            "rms_sample": rms.reshape(runs, count).T.tolist(),
            "relative_l2_sample": relative.reshape(runs, count).T.tolist()}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_cfg, model_cfg, operator_cfg, scheduler = build_configs(args)
    dataset = get_dataset(**data_cfg, device="cpu")
    n = len(dataset)
    first_id = int(data_cfg.get("start_id") or 0)
    if operator_cfg["name"] != "phase_retrieval":
        operator_cfg["device"] = device
    operator = get_operator(**operator_cfg)
    model = _load_prior(model_cfg, device).eval().requires_grad_(False)
    latent = model_cfg["name"] == "ldm"
    sampler = Pivot(scheduler, latent=latent, J=args.J, sigma_y=args.sigma_y, control=args.control, clip_grad=args.clip_grad)
    evaluator = Evaluator([get_eval_fn(name.strip()) for name in args.eval_fn_list.split(",") if name.strip()])

    root = Path(args.save_dir) / args.name
    root.mkdir(parents=True, exist_ok=True)
    ckpt = model_cfg["diffusion_path"] if latent else model_cfg["model_config"]["model_path"]
    with (root / "config.yaml").open("w") as f:
        yaml.safe_dump({"method": "PIVOT", "args": dict(vars(args), component_index_offset=first_id), "data": data_cfg,
                        "model": {"name": model_cfg["name"], "checkpoint": str(ckpt)}, "operator": operator_cfg,
                        "scheduler": scheduler}, f)
    image_dir = safe_dir(root / "samples")
    image_batches, measurements, runs, diagnostics = [], [], [], []
    for run in range(args.num_runs):
        run_samples = []
        for batch, start in enumerate(range(0, n, args.batch_size)):
            end = min(start + args.batch_size, n)
            if run == 0:
                images = torch.stack([dataset[i] for i in range(start, end)]).to(device)
                y = operator.measure(images)
                image_batches.append(images.cpu())
                measurements.append(y.cpu())
            else:
                y = measurements[batch].to(device)
            # Draw each start immediately after its measurement. Later runs reuse y.
            x_start = sampler.get_start(end - start, model)
            out = sampler.sample(model, x_start, operator, y)
            for j, img in enumerate(tensor_to_pils(out)):
                img.save(image_dir / f"{start + j:05d}_run{run:04d}.png")
            run_samples.append(out.cpu())
            ids = list(range(first_id + start, first_id + end))
            diagnostics.append({"run_id": run, "image_ids": ids, **sampler.diagnostics})
            print(f"Run {run}, images {ids[0]}-{ids[-1]}: {sampler.diagnostics['sampling_time_sec']:.3f} s", flush=True)
        runs.append(torch.cat(run_samples))

    images = torch.cat(image_batches).to(device)
    y = torch.cat(measurements).to(device)
    samples = torch.stack(runs).to(images)
    with torch.no_grad():
        results = evaluator.report(images, y, samples)
    table = evaluator.display(results)
    key = evaluator.main_eval_fn_name
    cmp = get_eval_fn_cmp(key)
    scores = np.asarray(results[key]["sample"])
    best = scores.argmin(axis=1) if cmp == "min" else scores.argmax(axis=1)
    results["best_run_idx"] = {"metric": key, "cmp": cmp, "index": best.tolist()}
    results["measurement_fidelity"] = _fidelity(operator, y, samples)
    results["sampling_diagnostics"] = diagnostics
    (root / "metrics.json").write_text(json.dumps(results, indent=4))
    (root / "sampling_diagnostics.json").write_text(json.dumps(diagnostics, indent=4))
    (root / "eval.md").write_text(table)
    grid = torch.cat([images, resize(y, images, operator.name), samples.flatten(0, 1)])
    grid = ((grid * 0.5 + 0.5).clamp(0, 1) * 255).round().div_(255)
    save_image(grid, root / "grid_results.png", nrow=n, padding=0)
    seconds = sum(item["sampling_time_sec"] for item in diagnostics) / (n * args.num_runs)
    print(table)
    print(f"Sampling: {seconds:.3f} s/sample")


if __name__ == "__main__":
    main()
