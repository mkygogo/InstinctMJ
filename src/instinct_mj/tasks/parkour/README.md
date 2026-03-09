# Parkour Task

Usage notes for the `InstinctMJ` parkour task family on `mjlab`.

## Basic Usage Guidelines

### Parkour Task

**Task IDs:**
- `Instinct-Parkour-Target-Amp-G1-v0` (train)
- `Instinct-Parkour-Target-Amp-G1-Play-v0` (play)

1. Go to `config/g1/g1_parkour_target_amp_cfg.py` and first update the local parkour dataset root:

   ```python
   _PARKOUR_DATASET_DIR = os.path.expanduser("~/Xyk/Datasets/data&model/parkour_motion_reference")
   ```

   `AmassMotionCfg.path` uses `_PARKOUR_DATASET_DIR` directly. If your filtered motion list is not stored as `parkour_motion_without_run.yaml` under that directory, also update `filtered_motion_selection_filepath` in the same file.

2. Train the policy:
```bash
instinct-train Instinct-Parkour-Target-Amp-G1-v0
```

3. Play trained policy (`--load-run` must be provided; absolute path is recommended. To visualize an untrained policy, use `--agent random`):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name>
```

4. Export trained policy (`--load-run` must be provided, absolute path is recommended):

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --export-onnx
```

5. Use the exported ONNX policy for play:

```bash
instinct-play Instinct-Parkour-Target-Amp-G1-Play-v0 --load-run <run_name> --use-onnx
```

## Common Options

- `--num-envs`: Number of parallel environments (default varies by task)
- `--load-run`: Run name to load checkpoint from for playing
- `--video`: Record training/playback videos
- `--export-onnx`: Export the trained model to ONNX format for onboard deployment during playing
- `--use-onnx`: Use the ONNX model for inference during playing
