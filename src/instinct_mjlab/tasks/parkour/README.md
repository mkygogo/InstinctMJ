# Parkour Task

## Task IDs

- `Instinct-Parkour-Target-Amp-G1-v0`
- `Instinct-Parkour-Target-Amp-G1-Play-v0`

## Dataset Configuration

Parkour supports two motion-input modes:

1. Pass `--motion-file /path/to/motion.npz` explicitly.
2. InstinctLab-style config: set the following fields in
   `tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py`:
   - `AmassMotionCfg.path`
   - `AmassMotionCfg.filtered_motion_selection_filepath`

When mode (2) is used, `instinct-train` / `instinct-play` will resolve motion data from this config.

`motion.npz` must contain:

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

## Train / Play

Train:

```bash
instinct-train Instinct-Parkour-Target-Amp-G1-v0 --env.scene.num-envs 2048
```

Train with explicit motion file:

```bash
instinct-train Instinct-Parkour-Target-Amp-G1-v0 \
  --motion-file /absolute/path/to/motion.npz \
  --env.scene.num-envs 2048
```

Play:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --checkpoint-file /absolute/path/to/model_x.pt \
  --viewer native
```

Play with explicit motion file:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 \
  --motion-file /absolute/path/to/motion.npz \
  --checkpoint-file /absolute/path/to/model_x.pt \
  --viewer native
```
