# TEACH: Temporal Variance-Driven Curriculum for Reinforcement Learning

Reference implementation of **TEACH**, a teacher–student framework for
goal-conditioned reinforcement learning in which the *temporal variance* of the
critic's value estimates drives adaptive goal selection.

> Gaurav Chaudhary, Laxmidhar Behera.
> *TEACH: Temporal Variance-Driven Curriculum for Reinforcement Learning.*
> International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2026. **(Oral)**

## Method

Uniform goal sampling is wasteful in multi-goal settings: most sampled goals are
either already solved or far out of reach. TEACH instead keeps a short rolling
history of the value estimate for each candidate goal and prioritises goals whose
estimates are still *moving*, measured as the standard deviation over that
history. Goals with high temporal variance are the ones the policy is actively
making progress on, so they carry the most learning signal.

The approach is algorithm-agnostic and attaches to an existing off-policy
goal-conditioned learner. The paper evaluates it on 11 robotic manipulation and
maze navigation tasks.

The core of the method is small and self-contained:

| File | Role |
|---|---|
| `goal_sampler.py` | **The contribution.** Tracks value history per goal (`val_track`) and weights candidate goals by temporal variance. |
| `curriculum.py` | Wires the sampler to an environment; builds both the variance-driven and uniform samplers for comparison. |
| `goal_env.py` | Gym wrapper letting the sampler override the environment's goal. |
| `agent.py` | Off-policy actor–critic learner. |
| `buffer.py` | Replay buffer with hindsight experience replay. |
| `train.py` | Training and evaluation loop. |
| `run.py` | CLI entry point. |

## Install

```bash
git clone https://github.com/gaurav-gaurav/teach.git
cd teach
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.9+ and PyTorch with CUDA. `--device cpu` works but is slow.

## Usage

```bash
# Fetch manipulation
python run.py --env_id FetchPickAndPlace-v2 --seed 1

# Uniform-goal baseline, for the comparison reported in the paper
python run.py --env_id FetchPickAndPlace-v2 --seed 1 --strategy uniform

# Maze navigation — requires the maze env, see Attribution below
python run.py --env_id MazeA-v0 --seed 1
```

Manipulation environments come from
[gymnasium-robotics](https://robotics.farama.org/). The maze tasks need one
extra step — see [Attribution](#attribution).

Runs log to [Weights & Biases](https://wandb.ai/) under project `teach_<env_id>`.
Use `wandb offline` to disable uploading.

### Key arguments

| Argument | Default | Meaning |
|---|---|---|
| `--env_id` | — | Gym environment id (required) |
| `--seed` | `1` | Random seed |
| `--freq` | `1` | How often the goal sampler is refreshed |
| `--val_history` | see `run.py` | Length of the value history used for the variance signal |
| `--strategy` | `variance` | Goal-scoring rule; `uniform` gives the baseline |
| `--num_timesteps` | `20000` | Total environment steps |
| `--num_seed_steps` | `1000` | Random-policy warmup steps |
| `--batch_size` | `128` | Batch size (includes HER samples) |
| `--lr` | `6e-4` | Actor and critic learning rate |
| `--discount` | `0.95` | Discount factor |
| `--hidden_sizes` | `[512, 512, 512]` | Policy and critic hidden layers |
| `--num_eval_episodes` | `5` | Episodes per evaluation |
| `--device` | `cuda` if available | Torch device |

`python run.py --help` lists all options.

## Attribution

All code here is the authors' own. Two things deliberately live outside it:

**Maze environment.** The particle-maze tasks use `ParticleMazeEnv` from the
[VDS codebase](https://github.com/zzyunzhi/vds) (`baselines/envs/maze/`), which
is not redistributed here. Copy `maze.py` and `maze_layouts.py` from that
repository into this directory to run `MazeA-v0` / `MazeB-v0` / `MazeC-v0`.
The manipulation tasks need nothing extra.

**Real-robot control.** The Kinova Gen3 code used for physical experiments
beyond the paper's scope is not part of this release; the reported results are
simulation-only.

## Scope and status

This is research code released to support the paper, not a maintained library.
It is the code used for the experiments, tidied for release: experiment logs,
checkpoints and scratch scripts removed, dead code stripped, and the curriculum
wiring rewritten so the repository contains only the authors' own code. No
algorithmic changes were made.

Reproducing the exact numbers in the paper needs multiple seeds per task; single
runs will vary.

## Citation

```bibtex
@inproceedings{chaudhary2026teach,
  title     = {TEACH: Temporal Variance-Driven Curriculum for Reinforcement Learning},
  author    = {Chaudhary, Gaurav and Behera, Laxmidhar},
  booktitle = {International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
