# BeyondMimic Task

This directory contains the `InstinctMJ` BeyondMimic task for whole-body tracking on `mjlab`, following the BeyondMimic setup (https://github.com/HybridRobotics/whole_body_tracking).

## Structure

```
beyondmimic/
├── __init__.py                   # Main module exports
├── beyondmimic_env_cfg.py        # Base environment configuration
├── README.md                     # This file
├── config/                       # Robot-specific configurations
│   ├── __init__.py
│   └── g1/                       # G1 robot configurations
│       ├── __init__.py
│       ├── beyondmimic_plane_cfg.py  # G1 plane environment config
│       ├── rl_cfgs.py            # Instinct-RL runner wiring
│       └── agents/               # Agent configurations
│           ├── __init__.py
│           └── beyondmimic_ppo_cfg.py  # PPO agent config
```

## Key Features

### BeyondMimic Approach
- **Link-level tracking**: Focuses on tracking individual body links rather than just joint positions
- **Relative world frame**: Uses relative world frame for link position and rotation tracking
- **Gaussian rewards**: Implements Gaussian-based reward functions for smooth tracking
- **Adaptive weighting**: Includes curriculum learning with adaptive motion weighting

### Reward Structure
The BeyondMimic reward system includes:
- Base position imitation (Gaussian)
- Base rotation imitation (Gaussian)
- Link position imitation (Gaussian, relative world frame)
- Link rotation imitation (Gaussian, relative world frame)
- Link linear velocity imitation (Gaussian)
- Link angular velocity imitation (Gaussian)
- Action rate regularization
- Joint limit penalties
- Undesired contact penalties

## Usage

### Configure Motion Source

Go to `config/g1/beyondmimic_plane_cfg.py` and update the motion source here:

```python
MOTION_NAME = "LafanSprint1"
_hacked_selected_file_ = "sprint1_subject2_retargetted.npz"
path=os.path.expanduser("~/Xyk/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz")
```

- `MOTION_NAME`: An identifier for the motion setup you are using.
- `_hacked_selected_file_`: The filename of the motion you want to use, relative to the dataset root.
- `path=os.path.expanduser(...)`: The local dataset root you need to change on your machine.
- `filtered_motion_selection_filepath`: This file is auto-generated from `MOTION_NAME`, so usually you do not need to edit it by hand.

### Training

```bash
instinct-train Instinct-BeyondMimic-Plane-G1-v0
```

### Playing Trained Policies

```bash
instinct-play Instinct-BeyondMimic-Plane-G1-Play-v0 --load-run <run_name>
```

To visualize an untrained policy, use `--agent random`.

### Python Configuration Access

```python
from instinct_mj.tasks.shadowing.beyondmimic.config.g1.beyondmimic_plane_cfg import g1_beyondmimic_plane_env_cfg
from instinct_mj.tasks.shadowing.beyondmimic.config.g1.rl_cfgs import g1_beyondmimic_instinct_rl_cfg

# Create environment configuration
env_cfg = g1_beyondmimic_plane_env_cfg(play=False)

# Create agent configuration
agent_cfg = g1_beyondmimic_instinct_rl_cfg()
```

## Implementation Notes

This task is organized around the current `InstinctMJ` / `mjlab` layout:

1. **Task Registration**: Registers train/play tasks through `register_instinct_task()`
2. **Environment Factory**: Uses `g1_beyondmimic_plane_env_cfg(play=...)` to switch between train and play setup
3. **Manager Style**: Keeps manager configuration in native `mjlab` / `InstinctMJ` terms
4. **Asset Integration**: Uses the G1 MJCF asset and `instinct_mj` motion reference stack
5. **Training Workflow**: Runs through the shared `instinct-train` / `instinct-play` entry points

## Motion Data

The configuration is set up to use the Ubisoft LAFAN-1 dataset with GMR retargeting for the G1 robot. By default, the current config points to:
```
~/Datasets/UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz
```

## References

- BeyondMimic: https://github.com/HybridRobotics/whole_body_tracking
- LAFAN-1 Dataset: https://github.com/ubisoft/ubisoft-laforge-animation-dataset
