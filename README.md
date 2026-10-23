# SuperMimic reproduction on mjlab (MuJoCo-Warp)

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model* (arXiv 2610.28480) — no official code is released, so everything here is implemented from
the paper. Target experiment: the controlled LAFAN1 comparison (Fig. 4a): 40 G1-retargeted LAFAN1
clips (2.45 h), one GPU; the paper reaches evaluation normalized return 0.82 after 1.2 h on an H100.
Originally written on Isaac Lab 3 (kit-less, Newton + MuJoCo-Warp; last version: commit `a1508d4`),
now running on [mjlab](https://github.com/mujocolab/mjlab) 1.6 with a plain uv environment.

Author: **Chong Zhang** (ETH Zurich). License: [MIT](LICENSE).

> **Disclaimer:** this repository was speed-vibecoded in 1 hour with an AI coding
> assistant. It
> deliberately does not live under any lab's organization, because the code is too much "slop"
> for that: fast and unreviewed. Use it at
> your own risk.

> **One G1 motion tracker for all 40 LAFAN1 clips, trained in only ~1.5 GPU-hours**
> (1.47 h on a single RTX PRO 6000, Isaac Lab version): evaluation return **0.829** with **0/40 failed clips**.
> For scale, the paper's PPO-in-simulator baseline does not reach return 0.82 within 18 h on up to
> 8 GPUs.

![SuperMimic G1 tracking LAFAN1 motions](results/supermimic_lafan1_highlight.gif)

*Solid robot: the trained policy simulated in Isaac Lab with MuJoCo-Warp (before the port to mjlab).
Blue ghost: the reference motion. Clips: dance, jumps, fall-and-get-up, sprint.*

Progress and timestamps: [MILESTONES.md](MILESTONES.md).

## Layout

| Path | What |
|---|---|
| `pyproject.toml`, `uv.lock` | uv project: mjlab 1.6.0, PyTorch < 2.11 and warp-lang < 1.18 (CUDA 12 builds) — see Setup |
| `data/` | not tracked; `data/lafan1_g1/` holds the 40 LAFAN1 G1 retargets (`tools/download_data.sh` fetches them from HF `lvhaidong/LAFAN1_Retargeting_Dataset`, or link a local copy) |
| `supermimic/` | implementation (see below) |
| `tools/` | tests, benchmark, plotting, rendering, failure analysis |
| `logs/<run>/` | not tracked: `train_log.csv`, `eval.csv`, checkpoints, stdout |
| `results/` | success curve and highlight GIF of the Isaac Lab run (full videos are rendered locally with `tools/render.py`) |

## Setup

**Requirements:** Linux, an NVIDIA GPU with driver >= 550 (CUDA 12.4), [uv](https://docs.astral.sh/uv/), and
`ffmpeg` (rendering only). Tested: RTX 6000 Ada, driver 550.163.01 (CUDA 12.4), uv 0.6.12, Python 3.12.

Physics: [mjlab](https://github.com/mujocolab/mjlab) 1.6.0 (`Scene` + `Simulation`, MuJoCo-Warp 3.11),
no Isaac Lab; 50 Hz control, 4 substeps of 5 ms, implicitfast integrator, pyramidal friction cones,
MuJoCo's Newton constraint solver with 10 iterations / 20 line-search iterations (as in mjlab's G1 tracking task), CUDA
graphs for step/forward/reset.

### 1. Environment

```bash
uv sync
```

Pins that keep everything on a CUDA 12.4 driver. CUDA 12.x builds run on it through CUDA minor
version compatibility; CUDA 13 builds need driver >= 580 and see no GPU (torch: "The NVIDIA driver on
your system is too old"; Warp: "Insufficient CUDA driver version"):

* **torch < 2.11**: from 2.11 on, the PyPI wheels are CUDA 13 builds; 2.8-2.10 are CUDA 12.8 builds
  (mjlab needs >= 2.7, and the last `cu124` build is 2.6). Resolves to torch 2.10.0; `torch.compile`
  (Triton 3.6) works on this driver. A newer torch with CUDA 12 comes from the PyTorch index instead:
  add `[[tool.uv.index]]` `name = "pytorch-cu126"`, `url = "https://download.pytorch.org/whl/cu126"`,
  `explicit = true`, set `[tool.uv.sources] torch = { index = "pytorch-cu126" }` and drop the upper
  bound (torch 2.14.1+cu126 passed the same checks).
* **warp-lang < 1.18**: from 1.18 on, the PyPI wheels are CUDA 13 builds; 1.14-1.17 are CUDA 12.9
  builds. MuJoCo-Warp's conditional CUDA graphs need driver >= 12.4, which this is.

### 2. Data: LAFAN1 motions

| What | Source | License | Goes to |
|---|---|---|---|
| 40 LAFAN1 clips retargeted to the 29-DoF G1 (30 fps CSV: root pos, root quat xyzw, 29 joint angles) | HF [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), folder `g1/` | CC BY-NC-ND 4.0 (Ubisoft La Forge) | `data/lafan1_g1/*.csv` |

```bash
tools/download_data.sh                                    # or, with a local copy:
ln -s /path/to/LAFAN1_Retargeting_Dataset/g1 data/lafan1_g1
```

The robot is mjlab's G1 (`g1_29dof_rev_1_0`, the revision the clips were retargeted to), so no
URDF/USD conversion is needed. `--motion_dir` also accepts a directory of mjlab motion NPZs
(`mjlab.scripts.csv_to_npz` output); their joint and body order is checked against the G1 FK on load.

Check: `ls data/lafan1_g1 | wc -l` prints `40`, and the clips total 264,705 frames at 30 fps
(2.45 h). `tools/test_env.py` checks the simulator against the reference motions (state round-trip
exact, FK vs. simulator link poses/velocities agree to ~1e-6):

```bash
uv run python tools/test_env.py
```

## Usage

```bash
CPUS=0-23 ./run_train.sh lafan1_main          # training in tmux session 'lafan1_main' (optional CPU pinning)
tmux attach -t lafan1_main
tmux new -d -s eval_main "uv run python supermimic/evaluate.py --run_dir logs/lafan1_main --watch"
uv run python tools/plot.py logs/lafan1_main --out results/success_curve.png
uv run python supermimic/evaluate.py --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_014000.pt --record traj.npz
uv run python tools/render.py traj.npz out.mp4 --clips dance2_subject1 --seconds 10
```

The first iterations are slow while `torch.compile` and MuJoCo-Warp build their kernels (cached in
`.cache/inductor` and `~/.cache/warp`). Checkpoints are written every 2000 iterations to
`logs/<run>/`; the eval watcher scores each on all 40 clips and appends to `logs/<run>/eval.csv`.

## Result (LAFAN1, 40 clips)

### mjlab 1.6 port (one RTX 6000 Ada, driver 550 / CUDA 12.4, torch 2.10)

Verification run of 4000 iterations (~1.9 it/s in the policy phase, eval watcher running alongside),
compared with the Isaac Lab run at the same iterations:

| Iteration | Train time | Eval return (mjlab) | Failed clips (mjlab) | Isaac Lab run |
|---|---|---|---|---|
| 2000 | 0.31 h | 0.544 | 13/40 | 0.541, 12/40 |
| 4000 | 0.62 h | 0.732 | 4/40 | 0.685, 4/40 |

Simulator throughput: ~550k control steps/s at 4096 environments (`tools/test_spawn.py`).

### Isaac Lab version (one RTX PRO 6000 Blackwell)

Evaluation return 0.824 at 1.24 h of training (paper: 0.82 at 1.2 h on an H100) and 0.829 with
**0/40 failed clips** at 1.47 h (iteration 14000).

![success curve](results/success_curve.png)

## Implementation (paper section -> file)

* G1 29-DoF (mjlab's `g1_29dof_rev_1_0` MJCF), BeyondMimic actuators as in mjlab's `G1_ARTICULATION`
  (10 Hz natural frequency, damping ratio 2, motor armatures, action scale 0.25·τmax/Kp) —
  `supermimic/robot.py`.
* Reference library: 30 -> 50 Hz resampling, finite-difference velocities, FK of 14 tracked bodies,
  1 s bins — `supermimic/motion.py`, differentiable torch FK `supermimic/kinematics.py` built from the
  simulated MJCF (matches the simulator's link poses/velocities to 1e-6).
* MDP (App. A): observation (q−q0, q̇, IMU ω, projected gravity, a_{t−1}; 10-frame history; Table A.1
  noise), K=10 future reference frames as the goal, BeyondMimic tracking reward without the
  self-collision term and joint-limit weight −2, termination |e_z|>0.25 m or e_tilt>0.8, sim context
  ξ = (friction U[0.3,1.2], joint offsets U[±0.01], torso CoM offset) — `supermimic/mdp.py`.
* Simulator wrapper on mjlab's `Scene`/`Simulation` (state read from the pelvis free joint's
  qpos/qvel; friction and torso CoM as per-world model fields) with return-based reset sampling
  (50% uniform / 40% softmax(−25 G_b/G_max) / 10% past-softmax) — `supermimic/env.py`.
* World model f_ω: MLP(2048,2048,2048) on heading-invariant state features + action + ξ, predicting
  frame-local increments; fit on k=8-step autoregressive windows (Eq. 1); analytic reward,
  observation and termination heads — `supermimic/networks.py`, `supermimic/train.py`.
* Policy: motion encoder MLP(2048,1024,512,256)->z∈R^32 (tanh), shared decoder
  MLP(2048,1024,512,256); critic MLP(256,128). BPTT over H=32 imagined steps from replay-buffer
  states with a soft survival weight and critic bootstrap (Eq. 2); critic on GAE(λ=0.95) targets
  (Eq. 3), 5 epochs × 4 minibatches.
* Hyperparameters as Table A.5: 4096 envs, buffer 1000 steps/env, action noise N(0, 0.1²),
  UTD 0.4 (2.5 vectorized sim steps per iteration), 10 WM batches of 2048 windows per iteration,
  4096 imagined rollouts, Adam lr 2e-4 (policy β=(0.9,0.95)), cosine decay to 0.03× over 250k
  iterations, grad clip 1.0 (+ element-wise 0.1 for the policy), γ=0.99.
* Evaluation (App. F.1): each clip once from its first frame, friction 1.0, no DR/noise,
  deterministic policy; failure = training termination — `supermimic/evaluate.py`.

## Deviations / choices not specified by the paper

* **Self-collisions disabled** in the simulator (speed); robot–ground contacts use the collision
  primitives of mjlab's G1 (capsules, spheres; the Isaac Lab version used the URDF's convex mesh
  hulls), all with condim 3 so that the randomized friction applies to every contact.
* **bf16 autocast for MLP matmuls** (paper: TF32). On an RTX PRO 6000 it is 24 % faster; all
  state integration, FK and rewards stay fp32.
* Reward, observation and termination heads are **analytic** (allowed by §3.2); termination in
  imagination is a sigmoid-softened version of the hard thresholds.
* The policy observation additionally contains the reference root orientation relative to the
  robot root (6D), i.e. BeyondMimic's `motion_anchor_ori_b` — IMU-observable; no global position.
* 500 world-model-only warm-up iterations before policy updates start.
* Torso-CoM and friction are randomized once per environment at start-up (BeyondMimic's
  start-up events); joint offsets are resampled every episode. Episodes time out after 10 s or at
  the clip end (bootstrapped, not terminal).
* Episode resets also clear MuJoCo-Warp's solver warm start and controls of the reset worlds
  (`Simulation.reset`).

## Attribution and licenses

* Method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
  CoRL 2026, arXiv:2610.28480. This code is an independent re-implementation, not affiliated with
  the authors.
* mjlab and MuJoCo-Warp (Apache-2.0) are used unmodified; the G1 MJCF and meshes ship with mjlab
  (derived from Unitree's G1 description, BSD-3-Clause). The Isaac Lab version used Isaac Lab
  (BSD-3-Clause) and Newton (Apache-2.0).
* LAFAN1 motions (Ubisoft La Forge): CC BY-NC-ND 4.0; G1 retargets by Unitree (BSD-3-Clause). The
  motion data is not redistributed here; the demo GIF shows the reference motions for illustration
  only and is not covered by this repository's MIT license.

## License

The code in this repository is released under the [MIT License](LICENSE).

Copyright (c) 2026 ETH Zurich, Chong Zhang.
