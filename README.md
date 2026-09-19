# UVTA

Co-training a visuo-tactile diffusion policy on **human** and **robot**
demonstrations. One wrist camera, one 20-D tactile vector, no proprioception;
the policy outputs a 31-D action (6-DoF wrist delta as xyz + rot6d, plus 22 hand
joint angles) and predicts future tactile from a separate regression head.

This repository is the minimal extract needed to reproduce the method:
training, offline evaluation, and the real-robot rollout. It is derived from
[DexUMI](https://github.com/real-stanford/DexUMI) (MIT); module and class names
under `uvta/` keep their upstream spelling so the lineage stays visible.

---

## Why a tactile head instead of a tactile output block

The obvious way to predict future tactile is to append it to the diffusion
target. Measured on 256 held-out anchors of the flip-book task, decoding the
*action* block identically in all three cases, that is a bad trade:

| diffusion target | wrist RMS x / y / z (mm) | chunk roughness (y) |
| --- | --- | --- |
| action only | 2.58 / 2.81 / 1.26 | 0.250 |
| action + tactile | 5.96 / 6.18 / 2.56 | 0.457 |
| action + state + tactile | 7.60 / 9.42 / 4.64 | 0.873 |
| *(teleop demos, reference)* | — | 0.197 |

One UNet denoises every dimension jointly for 16 steps, so residual uncertainty
in the tactile dims re-enters the action dims at every step. Note the training
loss does **not** show this: on epsilon-MSE the three sit at 2.35 / 3.14 / 3.04
mm and rank the wrong way round. Training loss scores one denoising step;
deployment runs 16 coupled ones.

UVTA therefore reads tactile off the (noise-free) conditioning vector with a
small MLP head. The gradient still flows through the shared visual trunk, so
the auxiliary supervision still shapes the representation, but the action's
reverse process never sees it.

```
                      +--------------- ViT-S/8 (wrist RGB) --+
observation ----------+                                      +-- cond (404-D)
                      +--------------- fsr_region (20-D) ----+      |
                                                            +-------+--------+
                                             diffusion UNet |                | MLP head
                                        (action, 31-D, 16   |                | (tactile,
                                         reverse steps)     |                |  16x20)
                                                            v                v
                                                     executed action   future tactile
```

---

## Install

```bash
git clone https://github.com/uni-vta/UVTA.git
cd UVTA

# Install the torch wheel matching your CUDA driver FIRST, then:
pip install -r requirements.txt
pip install -e .
```

Real-robot deployment needs more, including packages that ship with the
hardware rather than from PyPI -- see `requirements-deploy.txt`. Training and
offline evaluation need none of them.

Set the paths the configs are rooted on:

```bash
export UVTA_ROOT=/path/to/UVTA          # this clone
export UVTA_DATA_ROOT=/path/to/data     # defaults to $UVTA_ROOT/data
```

---

## Data

Each dataset is one zarr store of per-episode groups:

```
book_teleop/                 # robot, 149 episodes
  episode_0/
    camera_0/rgb    (T, 384, 480, 3) uint8   wrist camera
    camera_1/rgb    (T, 384, 480, 3) uint8   ego camera (unused by this config)
    pose            (T, 6)  float32          wrist pose, xyz + rotvec
    pose_action     (T, 6)  float32          commanded wrist pose
    proprioception  (T, 22) float32          hand joint angles, measured
    hand_action     (T, 22) float32          hand joint angles, commanded
    fsr             (T, 100) float32         raw taxels, 5 fingers x 20
    fsr_region      (T, 20) float32          <- what the policy consumes
    force           (T, 5)  float32          per-finger resultant
    tactile         (T, 5, 240, 240) uint8   raw deformation images
book_skeleton/               # human, 3074 episodes -- same layout, no `tactile`
```

`fsr_region` pools each finger's 20 taxels into 4 region means
(`deform_to_human/fsr_region.py` is the single source of truth for that
partition, and deployment re-applies it to the live stream). It is
**precomputed in the zarr**, so training never imports that module.

Two properties of the tactile stream are load-bearing and easy to get wrong:

- **Human and robot tactile are not on the same scale.** The human glove reads
  in raw sensor units (tens of thousands); the robot reads Newtons. Do not label
  them with the same units in a figure.
- **Resting offsets are removed per embodiment, differently.** The robot's rest
  is either a hard zero or a constant per-session DC offset, so subtracting its
  *first frame* estimates it exactly. The glove's rest is noisy (std ~2 within
  the window), so the human stream subtracts the *mean of 5 frames*. Both clamp
  at zero so `0` means "no contact" on both sides. See `fsr_baseline_*` in the
  config.

---

## Train

```bash
python scripts/train.py                  # single process
accelerate launch scripts/train.py       # multi-GPU
```

Everything comes from `configs/flip_book.yaml`; override on the command line in
Hydra syntax, e.g. `training.epochs=300 dataset.max_episode=[null,1000]`.

### The one setting that will bite you

`sampler: uniform` draws one epoch's worth of samples per epoch, so **steps per
epoch scale with dataset size** and a fixed epoch count buys very different
amounts of optimization as you vary the data:

| human episodes | samples | steps/epoch (global batch 4000) | 300 epochs |
| --- | --- | --- | --- |
| 3074 (all) | 738,517 | 184.6 | 55,389 updates |
| 1000 | 294,750 | 73.7 | 22,106 |
| 500 | 187,316 | 46.8 | 14,049 |
| 0 (robot only) | 41,947 | 10.5 | 3,146 |

That matters because budget alone is potent here: **23k updates gives 8.4 mm
sampled wrist RMS and 51k gives 1.6 mm**, with a steep cliff in between. If you
cut the data, raise `training.epochs` to keep the update count comparable -- and
keep `lr_scheduler_steps_epochs` equal to `training.epochs`, or the cosine is
truncated before it anneals.

---

## Evaluate offline

Training loss is a poor proxy (see the table at the top). This runs the real
reverse-diffusion sampler on held-out anchors and reports physical error:

```bash
python scripts/eval_action_quality.py \
    --runs <run_dir>:label [<run_dir>:label ...] \
    --data_dir $UVTA_DATA_ROOT/book_teleop --episodes 3 --anchors 256
```

It prints per-axis wrist RMS in millimetres plus chunk roughness against the
same statistic on the demonstrations -- a model jerkier than the demos will
visibly jitter no matter how good its average error is.

---

## Deploy

```bash
pip install -r requirements-deploy.txt
python cet/relative_policy_rollout.py --ckpt <run_dir> --epoch <n>
```

**This path is hardware-specific.** It talks to the arm through
`cet/north_env.py` over zenoh and solves IK with `north_kinematics`, which ships
with the robot rather than from PyPI. On different hardware, `north_env.py` is
the file to replace; `uvta/real_env/real_policy.py` above it is
hardware-agnostic (observation window in, unnormalized action chunk out) and
should not need changes.

`cet/`, `deform_to_human/` and `third_party/glove_retargeting/` resolve each
other by relative path at runtime, so keep them as siblings.

---

## Layout

```
uvta/                    importable library
  diffusion_policy/      model, UNet, datasets, replay buffer
  real_env/real_policy.py  checkpoint -> action chunk (no hardware)
  common/                EMA, zarr codecs, SE(3) helpers
configs/flip_book.yaml   the whole experiment
scripts/train.py         training entry point
scripts/eval_action_quality.py   offline sampled-action evaluation
cet/                     real-robot rollout (hardware-coupled)
deform_to_human/         live fsr -> fsr_region pooling + region maps
third_party/             vendored glove retargeting
assets/ring/             camera-ring inpainting mask for human frames
```

---

## Citation

```bibtex
@misc{uvta2026,
  title  = {UVTA: Co-training Visuo-Tactile Diffusion Policies from Human and Robot Demonstrations},
  year   = {2026},
  url    = {https://github.com/uni-vta/UVTA}
}
```

## License

MIT, see [LICENSE](LICENSE) -- including the third-party notices for DexUMI and
the vendored glove retargeting code.
