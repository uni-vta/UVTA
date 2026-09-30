# UVTA

**Unified Visual-Tactile-Action Modeling from Human Demonstrations for Dexterous Manipulation**

[Project website](https://uni-vta.github.io/) · [Dataset](https://huggingface.co/datasets/Chopper233/UVTA) · [Paper](https://arxiv.org/abs/2609.34182)

## Introduction

UVTA learns contact-aware dexterous manipulation from human tactile demonstrations
and robot teleoperation data. A shared visual-tactile representation conditions
a diffusion action head and an auxiliary future-tactile prediction head. Human
data supervise representation learning; robot actions are executed at deployment.

The default policy uses wrist RGB and a 20-D fingertip tactile vector to predict
16-step action chunks. Each action contains a 9-D relative wrist pose
(position + 6D rotation) and 22 hand joint targets. This repository provides
training, offline evaluation, and robot rollout code.

## Installation

Use Python 3.10 or later. Install PyTorch 2.5.1 and torchvision 0.20.1 for your
CUDA environment following the [PyTorch instructions](https://pytorch.org/get-started/previous-versions/), then:

```bash
git clone https://github.com/uni-vta/UVTA.git
cd UVTA
pip install -r requirements.txt
pip install -e .

export UVTA_ROOT="$PWD"
export UVTA_DATA_ROOT=/path/to/data
```

`UVTA_DATA_ROOT` defaults to `$UVTA_ROOT/data`. Datasets and checkpoints are
not included in this repository.

## Data

Download the released data from [Hugging Face](https://huggingface.co/datasets/Chopper233/UVTA).
It contains **Zarr v2 stores packaged into 28 `.tar` shards** (about 482 GB in
total), not Hugging Face Arrow tables or WebDataset samples. Download only the
tasks you need and allow space for both the archives and extracted data.

For example, download and extract all human/robot shards for `flip_book`:

```bash
pip install huggingface_hub
hf download Chopper233/UVTA --repo-type dataset \
    --include "flip_book/*" --include "MANIFEST.json" \
    --local-dir "$UVTA_ROOT/downloads"

mkdir -p "$UVTA_DATA_ROOT"
for archive in "$UVTA_ROOT"/downloads/flip_book/*.tar; do
    tar -xf "$archive" -C "$UVTA_DATA_ROOT"
done
```

Replace `flip_book` in both commands for another task. Extract **all shards of
each selected embodiment into the same data root**, preserving the archive paths
and hidden `.zgroup`/`.zarray` metadata; do not strip directory components or
create a separate directory per shard. `MANIFEST.json` lists the archive sizes,
SHA-256 checksums, and episode IDs. The loader reads extracted directories, not
`.tar` files, so point `UVTA_DATA_ROOT` at the directory containing the task
folders:

```text
<data_root>/<task>/
├── robot/
│   ├── episode_0/
│   └── ...
└── human/
    ├── episode_0/
    └── ...
```

Tasks: `flip_book`, `light_bulb`, `switch`, `ball`, and `tube`.
Each episode contains synchronized arrays:

```text
camera_0/rgb    (T, H, W, 3)  uint8 RGB
pose           (T, 6)        observed wrist position + rotation vector
pose_action    (T, 6)        target wrist position + rotation vector
proprioception (T, 22)       observed hand joint angles
hand_action    (T, 22)       target hand joint angles
tactile        (T, 20)       four tactile regions per fingertip
```

The released `tactile` field is already pooled into 20 regions; it is not a raw
tactile image or a 100-taxel vector and does not need another VTPM conversion.
Use the default `fsr_source_key: tactile`, `hand_action_mode: joint`, and
`load_camera_ids: [0]`. The loader converts the stored 6-D wrist poses to 9-D
relative wrist targets, giving 31-D actions together with the 22 hand joints.
Optional fingertip-pose targets and additional cameras are not in this release.

The manifest contains 1,000 human episodes per task and 150 robot episodes per
task, except `flip_book`, which contains 149 robot episodes. The replay buffer
loads selected episodes into RAM; use `'dataset.max_episode=[1,1]'` for an initial
small-data check before loading a full task.

Human and robot streams use separate normalization statistics. The loader
subtracts the first robot frame or the mean of the first five human frames
from tactile readings, then clamps at zero. Deployment applies the same robot
baseline correction. Region pooling is implemented in
[deform_to_human/fsr_region.py](deform_to_human/fsr_region.py).

## Training

[configs/flip_book.yaml](configs/flip_book.yaml) configures all five tasks:

```bash
python scripts/train.py
python scripts/train.py task=switch
python scripts/train.py task=light_bulb training.epochs=500
accelerate launch --multi_gpu scripts/train.py task=ball
```

The default is 300 epochs; the light-bulb experiment uses 500. The cosine
schedule follows `training.epochs`. With `training.sampler=uniform`, the
number of updates per epoch depends on dataset size; account for this when
comparing training budgets.

Runs are saved to `$UVTA_ROOT/experiments/<timestamp>`, or `UVTA_OUTPUT_DIR`
when set.

## Offline Evaluation

Evaluate sampled action chunks on held-out robot episodes:

```bash
python scripts/eval_action_quality.py \
    --runs /path/to/run:uvta \
    --data_dir /path/to/held-out/robot \
    --episodes 3 --anchors 256
```

The script reports wrist RMS error and chunk roughness in millimetres.
Use `/path/to/run:label@300` to select checkpoint 300; otherwise the latest
checkpoint is used.

## Robot Deployment

Deployment requires a hardware-specific `RobotEnv`, arm kinematics, and vendor
SDKs, which are not included. Adapt the observation and command mappings in
[cet/relative_policy_rollout.py](cet/relative_policy_rollout.py) and the interface
in [cet/robot_env.py](cet/robot_env.py) before running:

```bash
pip install -r requirements-deploy.txt
python cet/relative_policy_rollout.py --model_path /path/to/run --ckpt 300
```

Keep camera, hand, and tactile observations synchronized, and validate calibration
and motion limits before commanding a robot. The rollout applies tactile pooling
and the saved policy's baseline correction. Keep `cet/`, `deform_to_human/`,
and `third_party/` in their repository locations.

## Repository Layout

- `uvta/`: policy, datasets, normalization, and inference.
- `configs/`, `scripts/`: training configuration and entry points.
- `cet/`: robot rollout, observation timing, and trajectory interpolation.
- `deform_to_human/`: tactile conversion and region maps.
- `assets/`, `third_party/`: camera-ring assets and retargeting utilities.

## Citation

```bibtex
@misc{uvta2026,
  title = {Unified Visual-Tactile-Action Modeling from Human Demonstrations for Dexterous Manipulation},
  year  = {2026},
  url   = {https://uni-vta.github.io/}
}
```

## License

MIT; see [LICENSE](LICENSE) for copyright and third-party notices.
