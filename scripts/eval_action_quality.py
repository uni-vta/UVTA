"""Evaluate sampled action accuracy and temporal roughness on robot episodes.

Runs the reverse-diffusion sampler and decodes actions as in deployment.
Reports per-axis wrist RMS and mean absolute second differences in millimetres.

Usage and checkpoint selection are documented in README.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Where `scripts/train.py` writes runs; mirrors the config's
# UVTA_OUTPUT_DIR default so `--runs <timestamp>` just works.
_EXPERIMENTS = os.environ.get(
    "UVTA_OUTPUT_DIR", os.path.join(_ROOT, "experiments")
)

from uvta.common.utility.file import read_pickle  # noqa: E402
from uvta.common.utility.model import load_config, load_diffusion_model  # noqa: E402
from uvta.diffusion_policy.dataloader.diffusion_bc_dataset import (  # noqa: E402
    unnormalize_data,
)
from uvta.diffusion_policy.dataloader.stage1_rollout_runner import (  # noqa: E402
    apply_saved_stats,
    build_stage1_obs_dataset,
    sample_noise,
)
from uvta.real_env.real_policy import RealPolicy  # noqa: E402

_EEF_LEGACY = 6


@torch.no_grad()
def evaluate(run_dir, ckpt, data_dir, episodes, anchors, batch_size, seed,
             steps=None):
    path = os.path.join(_EXPERIMENTS, run_dir)
    cfg = load_config(path)
    use_ema = bool(cfg.training.use_ema)
    model, sched = load_diffusion_model(path, ckpt, use_ema=use_ema)
    model.eval().cuda()
    policy = RealPolicy(path, ckpt, use_ema=use_ema)

    # ``allow_unseen``: a HELD-OUT dataset is the whole point of a test-set run,
    # so the runner's "you never trained on this" guard has to be opted out of
    # here.  Normalization still comes from the run's own stats.pickle below, so
    # the model sees the scale it was trained with.
    ds = build_stage1_obs_dataset(cfg, data_dir, episodes, verbose=False,
                                  allow_unseen=True)
    apply_saved_stats(ds, read_pickle(os.path.join(path, "stats.pickle")), verbose=False)

    n = min(anchors, len(ds))
    idx = np.linspace(0, len(ds) - 1, n, dtype=np.int64)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, idx.tolist()),
        batch_size=batch_size, shuffle=False, num_workers=4,
    )
    A = int(np.asarray(policy.stats["action"]["min"]).size)
    H = int(cfg.dataset.pred_horizon)
    print(f"       [dbg] run={run_dir} blocks={policy.motor_blocks} "
          f"A(stats)={A} tactile={policy.tactile_action_dim} "
          f"ds_action_width={ds[0]['action'].shape[-1]}", flush=True)
    tk = str(cfg.dataset.get("fsr_source_key", "fsr"))
    cams = list(policy.camera_ids)

    pred_xyz, gt_xyz = [], []
    cursor = 0
    for batch in loader:
        B = batch["action"].shape[0]
        vis = torch.cat([batch[f"camera_{c}"].float().cuda() for c in cams], dim=1) if cams else None
        prop = batch["proprioception"].float().cuda() if "proprioception" in batch else None
        fsr = batch[tk].float().cuda() if tk in batch else None
        traj = sample_noise(idx[cursor:cursor + B], H, A, seed).cuda()
        cursor += B
        out = model.inference(
            proprioception=prop, fsr=fsr, visual_obs=vis, trajectory=traj,
            noise_scheduler=sched,
            num_inference_steps=int(steps or cfg.num_inference_steps),
        ).cpu().numpy()

        # Both prediction and ground truth take the SAME path: un-normalize,
        # drop tactile, keep the executed block, decode rot6d -> legacy.  Any
        # difference is then model error alone.
        def to_wrist(norm):
            act = unnormalize_data(norm.reshape(-1, A), policy.stats["action"]).reshape(B, H, A)
            if policy.tactile_action_dim:
                act = act[..., :-policy.tactile_action_dim]
            if len(policy.motor_blocks) > 1:
                act = np.stack([policy._extract_block(a, policy.executed_block) for a in act])
            dec = policy._decode_action_to_legacy(act.reshape(B * H, -1)).reshape(B, H, -1)
            return dec[..., :3]

        pred_xyz.append(to_wrist(out))
        gt_xyz.append(to_wrist(batch["action"].numpy()))

    p = np.concatenate(pred_xyz)          # (N, H, 3) relative wrist xyz, metres
    g = np.concatenate(gt_xyz)
    err = p - g
    rms = np.sqrt((err ** 2).mean(axis=(0, 1))) * 1000
    rough_p = np.abs(np.diff(p, n=2, axis=1)).mean(axis=(0, 1)) * 1000
    rough_g = np.abs(np.diff(g, n=2, axis=1)).mean(axis=(0, 1)) * 1000
    del model, policy
    torch.cuda.empty_cache()
    return dict(rms=rms, rough_p=rough_p, rough_g=rough_g, n=len(p),
                blocks=list(cfg.dataset.get("predict_state", False) and ["a", "s"] or ["a"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run_dir[:label][@ckpt] entries")
    ap.add_argument("--data_dir", default="data/book_teleop")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--anchors", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inference_steps", type=int, default=None,
                    help="override the checkpoint's num_inference_steps; the "
                         "deploy sampler budget is free to change at test time")
    args = ap.parse_args()

    results = {}
    for spec in args.runs:
        at = spec.split("@")
        head, ckpt = at[0], (int(at[1]) if len(at) > 1 else None)
        parts = head.split(":")
        run_dir, label = parts[0], (parts[1] if len(parts) > 1 else parts[0])
        if ckpt is None:
            ck = sorted(
                int(f.split("_")[1].split(".")[0])
                for f in os.listdir(os.path.join(_EXPERIMENTS, run_dir, "checkpoints"))
                if f.startswith("epoch_")
            )
            ckpt = ck[-1]
        print(f"[eval] {label}  ({run_dir} @ epoch {ckpt})", flush=True)
        results[label] = evaluate(run_dir, ckpt, args.data_dir, args.episodes,
                                  args.anchors, args.batch_size, args.seed,
                                  args.inference_steps)

    print()
    print("=" * 78)
    print(f"sampled with the real reverse diffusion, {args.data_dir}, "
          f"{results[list(results)[0]]['n']} anchors")
    print("=" * 78)
    print(f"{'model':26s} {'wrist RMS error (mm)':>26s} {'chunk roughness (mm)':>24s}")
    print(f"{'':26s} {'x':>8s}{'y':>9s}{'z':>9s} {'x':>8s}{'y':>8s}{'z':>8s}")
    for label, r in results.items():
        print(f"{label:26s} " + "".join(f"{v:9.3f}" for v in r["rms"])
              + "  " + "".join(f"{v:8.3f}" for v in r["rough_p"]))
    r0 = results[list(results)[0]]
    print(f"{'GROUND TRUTH roughness':26s} {'':27s}"
          + "".join(f"{v:8.3f}" for v in r0["rough_g"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
