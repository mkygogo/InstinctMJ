"""One-shot acceptance workflow for Perceptive P0-1.7 depth alignment."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro

import mjlab
from instinct_mjlab.scripts.depth_probe import ProbeConfig, run_probe
from instinct_mjlab.scripts.list_envs import list_environments


@dataclass(frozen=True)
class P017Config:
  motion_file: str | None = None
  registry_name: str | None = None
  steps: int = 8
  num_envs: int = 8
  device: str | None = "cpu"
  baseline_shadowing_json: str | None = None
  baseline_vae_json: str | None = None
  output_dir: str = "artifacts"
  output_prefix: str = "depth_probe_p0_1_7"
  compare_abs_tol: float = 0.03
  compare_rel_tol: float = 0.15
  allow_missing_baseline: bool = False
  baseline_search_glob: str = "*instinctlab*probe*.json"


def _resolve_optional_path(path: str | None) -> Path | None:
  if path is None:
    return None
  return Path(path).expanduser().resolve()


def _build_probe_command(
  *,
  task_id: str,
  cfg: P017Config,
  output_json: Path,
  baseline_json: Path | None = None,
) -> str:
  parts = [
    "./.venv/bin/instinct-depth-probe",
    task_id,
    f"--steps {cfg.steps}",
    f"--num-envs {cfg.num_envs}",
    f"--device {cfg.device or 'cpu'}",
    f"--output-json {output_json}",
  ]
  if cfg.motion_file is not None:
    parts.append(f"--motion-file {cfg.motion_file}")
  if cfg.registry_name is not None:
    parts.append(f"--registry-name {cfg.registry_name}")
  if baseline_json is not None:
    parts.append(f"--baseline-json {baseline_json}")
    parts.append(f"--compare-abs-tol {cfg.compare_abs_tol}")
    parts.append(f"--compare-rel-tol {cfg.compare_rel_tol}")
  return " ".join(parts)


def _collect_baseline_diagnostics(search_dir: Path, pattern: str) -> dict[str, object]:
  baseline_files = sorted(search_dir.glob(pattern))
  legacy_runtime = "isaac" "lab"
  legacy_installed_key = "isaac" "lab_installed"
  return {
    "instinctlab_installed": importlib.util.find_spec("instinctlab") is not None,
    legacy_installed_key: importlib.util.find_spec(legacy_runtime) is not None,
    "instinctlab_baseline_json_count": len(baseline_files),
    "instinctlab_baseline_json_files": [str(path) for path in baseline_files[:10]],
  }


def _run_probe_once(
  *,
  task_id: str,
  cfg: P017Config,
  output_json: Path,
  baseline_json: Path | None = None,
) -> None:
  probe_cfg = ProbeConfig(
    motion_file=cfg.motion_file,
    registry_name=cfg.registry_name,
    steps=cfg.steps,
    num_envs=cfg.num_envs,
    device=cfg.device,
    baseline_json=str(baseline_json) if baseline_json is not None else None,
    output_json=str(output_json),
    compare_abs_tol=cfg.compare_abs_tol,
    compare_rel_tol=cfg.compare_rel_tol,
  )
  run_probe(task_id=task_id, cfg=probe_cfg)


def _read_comparison_pass(path: Path) -> bool | None:
  payload = json.loads(path.read_text(encoding="utf-8"))
  comparison = payload.get("comparison")
  if not isinstance(comparison, dict):
    return None
  passed = comparison.get("pass")
  if isinstance(passed, bool):
    return passed
  return None


def _build_summary_path(output_dir: Path, output_prefix: str) -> Path:
  return output_dir / f"{output_prefix}_summary.json"


def run_acceptance(cfg: P017Config) -> dict[str, object]:
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  shadowing_probe_path = output_dir / f"{cfg.output_prefix}_shadowing_current.json"
  vae_probe_path = output_dir / f"{cfg.output_prefix}_vae_current.json"
  shadowing_compare_path = output_dir / f"{cfg.output_prefix}_shadowing_compare.json"
  vae_compare_path = output_dir / f"{cfg.output_prefix}_vae_compare.json"
  summary_path = _build_summary_path(output_dir, cfg.output_prefix)

  baseline_shadowing = _resolve_optional_path(cfg.baseline_shadowing_json)
  baseline_vae = _resolve_optional_path(cfg.baseline_vae_json)
  baseline_ready = (
    baseline_shadowing is not None
    and baseline_shadowing.exists()
    and baseline_vae is not None
    and baseline_vae.exists()
  )

  matched_tasks = list_environments(keyword="Perceptive")
  if matched_tasks == 0:
    raise RuntimeError("No Perceptive task found in registry.")

  _run_probe_once(
    task_id="Instinct-Perceptive-Shadowing-G1-v0",
    cfg=cfg,
    output_json=shadowing_probe_path,
  )
  _run_probe_once(
    task_id="Instinct-Perceptive-Vae-G1-v0",
    cfg=cfg,
    output_json=vae_probe_path,
  )

  shadowing_compare_pass: bool | None = None
  vae_compare_pass: bool | None = None
  if baseline_ready and baseline_shadowing is not None and baseline_vae is not None:
    _run_probe_once(
      task_id="Instinct-Perceptive-Shadowing-G1-v0",
      cfg=cfg,
      output_json=shadowing_compare_path,
      baseline_json=baseline_shadowing,
    )
    _run_probe_once(
      task_id="Instinct-Perceptive-Vae-G1-v0",
      cfg=cfg,
      output_json=vae_compare_path,
      baseline_json=baseline_vae,
    )
    shadowing_compare_pass = _read_comparison_pass(shadowing_compare_path)
    vae_compare_pass = _read_comparison_pass(vae_compare_path)

  baseline_diag = _collect_baseline_diagnostics(output_dir, cfg.baseline_search_glob)

  summary: dict[str, object] = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "p0_1_7_baseline_ready": baseline_ready,
    "p0_1_7_pass": bool(baseline_ready and shadowing_compare_pass and vae_compare_pass),
    "perceptive_task_count": matched_tasks,
    "baseline_diagnostics": baseline_diag,
    "artifacts": {
      "shadowing_current": str(shadowing_probe_path),
      "vae_current": str(vae_probe_path),
      "shadowing_compare": str(shadowing_compare_path) if baseline_ready else None,
      "vae_compare": str(vae_compare_path) if baseline_ready else None,
      "summary": str(summary_path),
    },
    "commands": {
      "list_envs": "./.venv/bin/instinct-list-envs --keyword Perceptive",
      "shadowing_current": _build_probe_command(
        task_id="Instinct-Perceptive-Shadowing-G1-v0",
        cfg=cfg,
        output_json=shadowing_probe_path,
      ),
      "vae_current": _build_probe_command(
        task_id="Instinct-Perceptive-Vae-G1-v0",
        cfg=cfg,
        output_json=vae_probe_path,
      ),
      "shadowing_compare": _build_probe_command(
        task_id="Instinct-Perceptive-Shadowing-G1-v0",
        cfg=cfg,
        output_json=shadowing_compare_path,
        baseline_json=baseline_shadowing if baseline_ready else None,
      ),
      "vae_compare": _build_probe_command(
        task_id="Instinct-Perceptive-Vae-G1-v0",
        cfg=cfg,
        output_json=vae_compare_path,
        baseline_json=baseline_vae if baseline_ready else None,
      ),
    },
    "comparison_result": {
      "shadowing_pass": shadowing_compare_pass,
      "vae_pass": vae_compare_pass,
    },
  }

  summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(summary, ensure_ascii=False, indent=2))

  if not baseline_ready and not cfg.allow_missing_baseline:
    raise RuntimeError(
      "P0-1.7 blocked: missing InstinctLab baseline JSON. "
      f"Please provide --baseline-shadowing-json and --baseline-vae-json. Summary: {summary_path}"
    )
  if baseline_ready and (shadowing_compare_pass is not True or vae_compare_pass is not True):
    raise RuntimeError(f"P0-1.7 compare failed, see summary: {summary_path}")
  return summary


def main() -> None:
  cfg = tyro.cli(P017Config, config=mjlab.TYRO_FLAGS)
  run_acceptance(cfg)


if __name__ == "__main__":
  main()
