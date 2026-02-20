from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import numpy as np
import os
from scipy import interpolate
import scipy.spatial.transform as tf
import torch
import trimesh
import uuid
import yaml
from typing import TYPE_CHECKING

try:
    import coacd
    _COACD_AVAILABLE = True
except ImportError:
    _COACD_AVAILABLE = False

# Cache CoACD decomposition results keyed by (abspath, params...) so the same
# STL is only decomposed once even when the terrain generator creates many tiles.
_COACD_PARTS_CACHE: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
_COACD_PREWARM_DONE: set[tuple] = set()
_COACD_CACHE_VERSION = 2
_COACD_LOG_LEVEL_SET: str | None = None

import mujoco

from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
from mjlab.terrains.height_field.utils import convert_height_field_to_mesh

from ..height_field.hf_terrains import generate_perlin_noise
from .utils import crop_terrain_mesh_aabb, generate_wall

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def _set_coacd_log_level(level: str) -> None:
    """Set CoACD logger level once per process."""
    global _COACD_LOG_LEVEL_SET
    if not _COACD_AVAILABLE:
        return
    level = str(level)
    if _COACD_LOG_LEVEL_SET == level:
        return
    if not hasattr(coacd, "set_log_level"):
        return
    candidate_levels = (level, "off", "error", "warn")
    for candidate in candidate_levels:
        try:
            coacd.set_log_level(candidate)
            _COACD_LOG_LEVEL_SET = candidate
            return
        except Exception:
            continue


def _top_surface_samples(
    terrain_mesh: trimesh.Trimesh,
    normal_z_threshold: float,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract top-surface samples from mesh for heightfield collision fitting."""
    face_mask = np.abs(terrain_mesh.face_normals[:, 2]) > normal_z_threshold
    if np.any(face_mask):
        vertex_indices = np.unique(terrain_mesh.faces[face_mask].reshape(-1))
        samples = terrain_mesh.vertices[vertex_indices]
    else:
        samples = terrain_mesh.vertices

    bucket = max(float(resolution) * 0.5, 1.0e-6)
    quantized_xy = np.round(samples[:, :2] / bucket).astype(np.int64)
    unique_xy, inverse_indices = np.unique(quantized_xy, axis=0, return_inverse=True)
    top_z = np.full(unique_xy.shape[0], -np.inf, dtype=np.float64)
    np.maximum.at(top_z, inverse_indices, samples[:, 2])

    sample_xy = unique_xy.astype(np.float64) * bucket
    return sample_xy, top_z


def _coacd_run_kwargs(cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg) -> dict:
    """Collect CoACD kwargs from config."""
    return {
        "threshold": float(cfg.collision_coacd_threshold),
        "max_convex_hull": int(cfg.collision_coacd_max_convex_hull),
        "preprocess_mode": str(cfg.collision_coacd_preprocess_mode),
        "preprocess_resolution": int(cfg.collision_coacd_preprocess_resolution),
        "resolution": int(cfg.collision_coacd_resolution),
        "mcts_nodes": int(cfg.collision_coacd_mcts_nodes),
        "mcts_iterations": int(cfg.collision_coacd_mcts_iterations),
        "mcts_max_depth": int(cfg.collision_coacd_mcts_max_depth),
        "pca": bool(cfg.collision_coacd_pca),
        "merge": bool(cfg.collision_coacd_merge),
        "decimate": bool(cfg.collision_coacd_decimate),
        "max_ch_vertex": int(cfg.collision_coacd_max_ch_vertex),
        "extrude": bool(cfg.collision_coacd_extrude),
        "extrude_margin": float(cfg.collision_coacd_extrude_margin),
        "apx_mode": str(cfg.collision_coacd_apx_mode),
        "seed": int(cfg.collision_coacd_seed),
    }


def _coacd_cache_key(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrain_abspath: str,
) -> tuple:
    """Build a stable cache key for CoACD decomposition."""
    terrain_abspath = os.path.abspath(terrain_abspath)
    stat_info = os.stat(terrain_abspath)
    return (
        _COACD_CACHE_VERSION,
        terrain_abspath,
        int(stat_info.st_size),
        int(stat_info.st_mtime_ns),
        float(cfg.size[0]),
        float(cfg.size[1]),
        float(cfg.collision_coacd_threshold),
        int(cfg.collision_coacd_max_convex_hull),
        str(cfg.collision_coacd_preprocess_mode),
        int(cfg.collision_coacd_preprocess_resolution),
        int(cfg.collision_coacd_resolution),
        int(cfg.collision_coacd_mcts_nodes),
        int(cfg.collision_coacd_mcts_iterations),
        int(cfg.collision_coacd_mcts_max_depth),
        bool(cfg.collision_coacd_pca),
        bool(cfg.collision_coacd_merge),
        bool(cfg.collision_coacd_decimate),
        int(cfg.collision_coacd_max_ch_vertex),
        bool(cfg.collision_coacd_extrude),
        float(cfg.collision_coacd_extrude_margin),
        str(cfg.collision_coacd_apx_mode),
        int(cfg.collision_coacd_seed),
    )


def _coacd_cache_path(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrain_abspath: str,
    cache_key: tuple,
) -> str:
    """Get disk-cache file path for CoACD decomposition."""
    terrain_abspath = os.path.abspath(terrain_abspath)
    terrain_dir = os.path.dirname(terrain_abspath)
    cache_dir = os.path.join(terrain_dir, str(cfg.collision_coacd_cache_dirname))
    key_hash = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    terrain_stem = os.path.splitext(os.path.basename(terrain_abspath))[0]
    return os.path.join(cache_dir, f"{terrain_stem}.{key_hash}.npz")


def _load_coacd_parts_from_disk(cache_path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load CoACD parts from a compressed npz cache file."""
    with np.load(cache_path, allow_pickle=False) as cache:
        if "num_parts" not in cache:
            raise ValueError(f"Invalid CoACD cache file (missing num_parts): {cache_path}")
        num_parts = int(np.asarray(cache["num_parts"]).reshape(-1)[0])
        parts_arrays = []
        for part_idx in range(num_parts):
            verts_key = f"verts_{part_idx}"
            faces_key = f"faces_{part_idx}"
            if verts_key not in cache or faces_key not in cache:
                raise ValueError(
                    f"Invalid CoACD cache file (missing part arrays): {cache_path}, "
                    f"part_idx={part_idx}"
                )
            parts_arrays.append(
                (
                    np.asarray(cache[verts_key], dtype=np.float32),
                    np.asarray(cache[faces_key], dtype=np.int32),
                )
            )
    return parts_arrays


def _save_coacd_parts_to_disk(cache_path: str, parts_arrays: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Save CoACD parts into a compressed npz cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {"num_parts": np.asarray([len(parts_arrays)], dtype=np.int32)}
    for part_idx, (verts, faces) in enumerate(parts_arrays):
        payload[f"verts_{part_idx}"] = np.asarray(verts, dtype=np.float32)
        payload[f"faces_{part_idx}"] = np.asarray(faces, dtype=np.int32)

    tmp_cache_path = f"{cache_path}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez_compressed(tmp_cache_path, **payload)
        os.replace(tmp_cache_path, cache_path)
    finally:
        if os.path.exists(tmp_cache_path):
            os.remove(tmp_cache_path)


def _compute_motion_matched_border_height(
    terrain_mesh: trimesh.Trimesh,
    size: tuple[float, float],
    terrain_file: str,
) -> float:
    """Compute border height offset for motion-matched terrain alignment."""
    # Find border height offset w.r.t current center of this terrain mesh.
    # This is used to align the terrain mesh with the origin.
    # NOTE: Assuming the border is flat, we take the mean height of the vertices
    verts = terrain_mesh.vertices
    border_mask = np.logical_or(
        np.abs(verts[:, 0]) > (size[0] / 2 - 0.05),
        np.abs(verts[:, 1]) > (size[1] / 2 - 0.05),
    )
    border_verts_z = verts[border_mask][:, 2]

    if len(border_verts_z) == 0:
        # No vertices at cfg.size boundary — mesh is smaller than cfg.size.
        # Fall back to 0.0, matching original InstinctLab behavior for these terrains.
        # These STL meshes were recorded with the walkable floor at z≈0 by convention,
        # so no z-shift is needed.  Using AABB edges as fallback is unsafe because for
        # staircase/ramp terrains the AABB edges sample obstacle tops (z=0.6–0.7 m),
        # which would incorrectly sink the entire tile by that amount.
        border_height = 0.0
        print(
            f"[TerrainImporter] Terrain {terrain_file} has no border verts at cfg.size edges; "
            f"using border_height=0.0 (floor assumed at z≈0)."
        )
    else:
        border_height = float(np.mean(border_verts_z))
        if np.isnan(border_height):
            # Unexpected — empty slice mean; treat as 0.
            print(f"Warning: Terrain {terrain_file} does not have a valid border height. Using 0 as the border height.")
            border_height = 0.0
    return border_height


def _load_motion_matched_terrain_mesh(
    terrain_abspath: str,
    terrain_file: str,
    size: tuple[float, float],
) -> tuple[trimesh.Trimesh, float]:
    """Load, crop, and align one terrain mesh to generator convention."""
    terrain_mesh = trimesh.load(terrain_abspath, force="mesh")

    # crop terrain mesh by cfg.size
    # This does not change the terrain origin.
    terrain_mesh = crop_terrain_mesh_aabb(
        terrain_mesh,
        x_max=size[0] / 2,
        x_min=-size[0] / 2,
        y_max=size[1] / 2,
        y_min=-size[1] / 2,
    )
    # Normalize mesh winding/normals before CoACD to reduce manifold-orientation artifacts.
    terrain_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(terrain_mesh)
    trimesh.repair.fix_normals(terrain_mesh, multibody=True)

    border_height = _compute_motion_matched_border_height(
        terrain_mesh=terrain_mesh,
        size=size,
        terrain_file=terrain_file,
    )

    # To follow the terrain_generator convention, we move the terrain mesh to (size[0]/2, size[1]/2, -border_height).
    move_terrain_transform = np.eye(4)
    move_terrain_transform[:2, 3] = np.asarray(size, dtype=np.float64) / 2.0
    move_terrain_transform[2, 3] = -border_height
    terrain_mesh.apply_transform(move_terrain_transform)
    return terrain_mesh, border_height


def _run_coacd_decomposition(
    terrain_mesh: trimesh.Trimesh,
    coacd_kwargs: dict,
    terrain_tag: str,
    coacd_log_level: str,
    verbose: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Execute CoACD and return numpy-array parts."""
    _set_coacd_log_level(coacd_log_level)

    coacd_mesh = coacd.Mesh(
        vertices=np.asarray(terrain_mesh.vertices, dtype=np.float64),
        indices=np.asarray(terrain_mesh.faces, dtype=np.int32),
    )

    if verbose:
        print(
            f"[TerrainImporter] CoACD start: {terrain_tag}, "
            f"th={coacd_kwargs['threshold']}, "
            f"max_hull={coacd_kwargs['max_convex_hull']}, "
            f"merge={coacd_kwargs['merge']}, "
            f"apx={coacd_kwargs['apx_mode']}."
        )

    # Run CoACD convex decomposition.
    raw_parts = coacd.run_coacd(
        coacd_mesh,
        **coacd_kwargs,
    )

    # Store numpy arrays in cache (not raw CoACD objects)
    parts_arrays = [
        (np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int32))
        for v, f in raw_parts
    ]
    if verbose:
        print(f"[TerrainImporter] CoACD done: {terrain_tag}, hulls={len(parts_arrays)}.")
    return parts_arrays


def _coacd_prewarm_worker(
    job: tuple[str, str, tuple[float, float], dict, tuple, str, str],
) -> tuple[str, str, int]:
    """Worker job for CoACD cache prewarm."""
    if not _COACD_AVAILABLE:
        raise ImportError(
            "CoACD is required for collision_coacd=True. "
            "Install it with: pip install coacd"
        )
    terrain_file, terrain_abspath, size, coacd_kwargs, cache_key, coacd_log_level, cache_path = job
    terrain_mesh, _ = _load_motion_matched_terrain_mesh(
        terrain_abspath=terrain_abspath,
        terrain_file=terrain_file,
        size=size,
    )
    parts_arrays = _run_coacd_decomposition(
        terrain_mesh=terrain_mesh,
        coacd_kwargs=coacd_kwargs,
        terrain_tag=f"terrain_file={terrain_file}",
        coacd_log_level=coacd_log_level,
        verbose=False,
    )
    _save_coacd_parts_to_disk(cache_path, parts_arrays)
    return terrain_abspath, cache_path, len(parts_arrays)


def _render_progress_bar(done: int, total: int, width: int = 24) -> str:
    """Render a one-line ASCII progress bar."""
    total = max(total, 1)
    done = int(np.clip(done, 0, total))
    filled = int(round(width * done / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_prewarm_progress(done: int, total: int) -> None:
    """Print CoACD prewarm progress in-place."""
    progress_bar = _render_progress_bar(done, total)
    percent = (100.0 * done / max(total, 1))
    remaining = max(total - done, 0)
    print(
        f"\r[TerrainImporter] CoACD prewarm {progress_bar} "
        f"{done}/{total} ({percent:5.1f}%) remaining={remaining}",
        end="",
        flush=True,
    )


def _prewarm_coacd_disk_cache(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrains: list[dict],
) -> None:
    """Precompute all required CoACD decompositions with compact progress output."""
    if not cfg.collision_coacd_prewarm_all:
        return

    prewarm_key = (
        os.path.abspath(cfg.path),
        os.path.abspath(cfg.metadata_yaml),
        float(cfg.size[0]),
        float(cfg.size[1]),
        float(cfg.collision_coacd_threshold),
        int(cfg.collision_coacd_max_convex_hull),
        str(cfg.collision_coacd_preprocess_mode),
        int(cfg.collision_coacd_preprocess_resolution),
        int(cfg.collision_coacd_resolution),
        int(cfg.collision_coacd_mcts_nodes),
        int(cfg.collision_coacd_mcts_iterations),
        int(cfg.collision_coacd_mcts_max_depth),
        bool(cfg.collision_coacd_pca),
        bool(cfg.collision_coacd_merge),
        bool(cfg.collision_coacd_decimate),
        int(cfg.collision_coacd_max_ch_vertex),
        bool(cfg.collision_coacd_extrude),
        float(cfg.collision_coacd_extrude_margin),
        str(cfg.collision_coacd_apx_mode),
        int(cfg.collision_coacd_seed),
        bool(cfg.collision_coacd_use_disk_cache),
        str(cfg.collision_coacd_cache_dirname),
    )
    if prewarm_key in _COACD_PREWARM_DONE:
        return

    size = (float(cfg.size[0]), float(cfg.size[1]))
    coacd_kwargs = _coacd_run_kwargs(cfg)
    coacd_log_level = str(cfg.collision_coacd_log_level)

    jobs_by_path: dict[str, tuple[str, str, tuple, str]] = {}
    for terrain_entry in terrains:
        terrain_file = terrain_entry["terrain_file"]
        terrain_abspath = os.path.join(cfg.path, terrain_file)
        cache_key = _coacd_cache_key(cfg, terrain_abspath)
        if _COACD_PARTS_CACHE.get(cache_key) is not None:
            continue
        cache_path = _coacd_cache_path(cfg, terrain_abspath, cache_key)
        if cfg.collision_coacd_use_disk_cache and os.path.exists(cache_path):
            continue
        # Deduplicate repeated terrain files in metadata.
        jobs_by_path[terrain_abspath] = (
            terrain_file,
            terrain_abspath,
            cache_key,
            cache_path,
        )

    jobs = list(jobs_by_path.values())
    if len(jobs) == 0:
        _COACD_PREWARM_DONE.add(prewarm_key)
        return

    worker_count = int(cfg.collision_coacd_prewarm_workers)
    if worker_count <= 0:
        worker_count = max(os.cpu_count() or 1, 1)
    worker_count = min(worker_count, len(jobs))

    total_jobs = len(jobs)
    _print_prewarm_progress(0, total_jobs)

    if worker_count == 1:
        for done_count, (terrain_file, terrain_abspath, cache_key, cache_path) in enumerate(jobs, start=1):
            terrain_mesh, _ = _load_motion_matched_terrain_mesh(
                terrain_abspath=terrain_abspath,
                terrain_file=terrain_file,
                size=size,
            )
            parts_arrays = _run_coacd_decomposition(
                terrain_mesh=terrain_mesh,
                coacd_kwargs=coacd_kwargs,
                terrain_tag=f"terrain_file={terrain_file}",
                coacd_log_level=coacd_log_level,
                verbose=False,
            )
            _COACD_PARTS_CACHE[cache_key] = parts_arrays
            if cfg.collision_coacd_use_disk_cache:
                _save_coacd_parts_to_disk(cache_path, parts_arrays)
            _print_prewarm_progress(done_count, total_jobs)
    else:
        worker_jobs = [
            (
                terrain_file,
                terrain_abspath,
                size,
                coacd_kwargs,
                cache_key,
                coacd_log_level,
                cache_path,
            )
            for terrain_file, terrain_abspath, cache_key, cache_path in jobs
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_coacd_prewarm_worker, worker_job) for worker_job in worker_jobs]
            done_count = 0
            for future in as_completed(futures):
                future.result()
                done_count += 1
                _print_prewarm_progress(done_count, total_jobs)

    _COACD_PREWARM_DONE.add(prewarm_key)
    print()


def _add_collision_face_boxes(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    spec: mujoco.MjSpec,
    terrain_mesh: trimesh.Trimesh,
) -> list[TerrainGeometry]:
    """Build surface-only collision for a closed mesh by placing a thin box geom on each face.

    Root cause: MuJoCo's polyhedral mesh collision model treats the interior of any
    closed mesh as solid.  Objects near the inner surface receive negative-dist contacts
    even when not physically penetrating, causing ``illegal_reset_contact`` to fire
    immediately after spawn.

    Fix: disable collision on the visual mesh geom and replace it with one thin
    ``mjGEOM_BOX`` per triangle face, placed on the outward side of the face.
    Each box only collides from one side, so the robot inside the room is free
    to move without spurious contacts.
    """
    thickness = float(cfg.face_box_thickness)
    geometries: list[TerrainGeometry] = []
    face_normals = terrain_mesh.face_normals  # [F, 3], unit normals

    for face, normal in zip(terrain_mesh.faces, face_normals):
        verts = terrain_mesh.vertices[face]  # [3, 3]
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            # Degenerate face — skip
            continue
        unit_normal = normal / norm_len

        # Face centroid
        center = verts.mean(axis=0)

        # Build a local frame: t1 along first edge, t2 = normal x t1
        edge = verts[1] - verts[0]
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-10:
            continue
        t1 = edge / edge_len
        t2 = np.cross(unit_normal, t1)

        # Project vertices onto the face plane to get half-extents
        proj = verts - center
        coords_t1 = proj @ t1
        coords_t2 = proj @ t2
        half_t1 = (coords_t1.max() - coords_t1.min()) / 2.0 + 0.01
        half_t2 = (coords_t2.max() - coords_t2.min()) / 2.0 + 0.01

        # Box center: offset outward by half-thickness so the inner face sits at the surface
        box_center = center + unit_normal * (thickness / 2.0)

        # Rotation matrix: columns are [t1, t2, normal] -> MuJoCo quaternion [w, x, y, z]
        rot = np.column_stack([t1, t2, unit_normal])
        mat4 = np.eye(4)
        mat4[:3, :3] = rot
        quat_wxyz = trimesh.transformations.quaternion_from_matrix(mat4)  # [w, x, y, z]

        geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half_t1, half_t2, thickness / 2.0],
            pos=box_center.tolist(),
            quat=quat_wxyz.tolist(),
        )
        # Collision-only: invisible in default render group
        geom.group = 3
        geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
        geometries.append(TerrainGeometry(geom=geom))

    print(
        f"[TerrainImporter] Face-box collision: "
        f"{len(geometries)} box geoms generated from {len(terrain_mesh.faces)} faces."
    )
    return geometries


def _top_surface_height_map(
    sample_xy: np.ndarray,
    sample_z: np.ndarray,
    resolution: float,
) -> dict[tuple[int, int], float]:
    """Build max-z map indexed by quantized XY bins."""
    bucket = max(float(resolution), 1.0e-6)
    quantized_xy = np.round(sample_xy / bucket).astype(np.int64)
    height_map: dict[tuple[int, int], float] = {}
    for i in range(quantized_xy.shape[0]):
        key = (int(quantized_xy[i, 0]), int(quantized_xy[i, 1]))
        z = float(sample_z[i])
        previous = height_map.get(key)
        if previous is None or z > previous:
            height_map[key] = z
    return height_map


def _compute_coacd_auto_align_z(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    terrain_mesh: trimesh.Trimesh,
    parts_arrays: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Compute auto z-alignment from visual top surface to CoACD top surface."""
    if len(parts_arrays) == 0:
        return 0.0

    align_resolution = float(cfg.collision_coacd_auto_align_resolution)
    mesh_xy, mesh_z = _top_surface_samples(
        terrain_mesh=terrain_mesh,
        normal_z_threshold=cfg.collision_hfield_normal_z_threshold,
        resolution=align_resolution,
    )
    mesh_map = _top_surface_height_map(mesh_xy, mesh_z, align_resolution)

    verts_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    vert_offset = 0
    for part_verts, part_faces in parts_arrays:
        verts = np.asarray(part_verts, dtype=np.float64)
        faces = np.asarray(part_faces, dtype=np.int64)
        verts_list.append(verts)
        faces_list.append(faces + vert_offset)
        vert_offset += verts.shape[0]
    coacd_mesh = trimesh.Trimesh(
        vertices=np.concatenate(verts_list, axis=0),
        faces=np.concatenate(faces_list, axis=0),
        process=False,
    )
    coacd_xy, coacd_z = _top_surface_samples(
        terrain_mesh=coacd_mesh,
        normal_z_threshold=cfg.collision_hfield_normal_z_threshold,
        resolution=align_resolution,
    )
    coacd_map = _top_surface_height_map(coacd_xy, coacd_z, align_resolution)

    common_keys = mesh_map.keys() & coacd_map.keys()
    if len(common_keys) == 0:
        return 0.0

    z_diff = np.asarray(
        [mesh_map[key] - coacd_map[key] for key in common_keys],
        dtype=np.float64,
    )
    if z_diff.size == 0:
        return 0.0
    return float(np.median(z_diff))


def _add_collision_coacd(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    spec: mujoco.MjSpec,
    terrain_mesh: trimesh.Trimesh,
    terrain_idx: int,
    terrain_abspath: str,
) -> list[TerrainGeometry]:
    """Build collision geometry for a concave mesh using CoACD approximate convex decomposition.

    Root cause: MuJoCo's polyhedral mesh collision model treats the interior of any
    closed mesh as solid.  A robot standing inside a 3D-scanned room receives
    negative-dist contacts even when not physically penetrating.

    Fix: decompose the mesh into approximate convex hulls with CoACD.  Each hull
    is registered as a separate MuJoCo mesh and added as a collision-only geom
    (group 3, invisible).  The original mesh geom is kept for rendering / depth
    ray-casting (group 2, no collision).

    Results are cached in memory and on disk so the same STL is only decomposed
    once across terrain tiles and across process restarts.
    """
    if not _COACD_AVAILABLE:
        raise ImportError(
            "CoACD is required for collision_coacd=True. "
            "Install it with: pip install coacd"
        )

    cache_key = _coacd_cache_key(cfg, terrain_abspath)
    cache_path = _coacd_cache_path(cfg, terrain_abspath, cache_key)
    coacd_kwargs = _coacd_run_kwargs(cfg)

    cached_parts = _COACD_PARTS_CACHE.get(cache_key)
    if cached_parts is not None:
        parts_arrays = cached_parts
    else:
        parts_arrays = None
        if cfg.collision_coacd_use_disk_cache and os.path.exists(cache_path):
            try:
                parts_arrays = _load_coacd_parts_from_disk(cache_path)
            except (OSError, ValueError) as exc:
                print(
                    f"[TerrainImporter] CoACD disk cache load failed, recomputing: "
                    f"{cache_path}, error={exc}"
                )

        if parts_arrays is None:
            parts_arrays = _run_coacd_decomposition(
                terrain_mesh=terrain_mesh,
                coacd_kwargs=coacd_kwargs,
                terrain_tag=f"terrain_idx={terrain_idx}",
                coacd_log_level=str(cfg.collision_coacd_log_level),
                verbose=False,
            )
            if cfg.collision_coacd_use_disk_cache:
                _save_coacd_parts_to_disk(cache_path, parts_arrays)
        _COACD_PARTS_CACHE[cache_key] = parts_arrays

    auto_align_z = 0.0
    if bool(cfg.collision_coacd_auto_align_top_surface):
        auto_align_z = _compute_coacd_auto_align_z(
            cfg=cfg,
            terrain_mesh=terrain_mesh,
            parts_arrays=parts_arrays,
        )

    collision_z = float(cfg.collision_coacd_z_offset) + auto_align_z

    geometries: list[TerrainGeometry] = []
    for part_idx, (part_verts, part_faces) in enumerate(parts_arrays):
        hull_mesh_name = f"motion_matched_coacd_t{terrain_idx}_h{part_idx}_{uuid.uuid4().hex}"
        spec.add_mesh(
            name=hull_mesh_name,
            uservert=part_verts.reshape(-1).tolist(),
            userface=part_faces.reshape(-1).tolist(),
        )
        hull_geom = spec.body("terrain").add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=hull_mesh_name,
            pos=(0.0, 0.0, collision_z),
        )
        hull_geom.group = 3
        hull_geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
        hull_geom.margin = float(cfg.collision_coacd_geom_margin)
        hull_geom.gap = 0.0
        geometries.append(TerrainGeometry(geom=hull_geom))

    return geometries


def _add_collision_hfield_from_mesh(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    spec: mujoco.MjSpec,
    terrain_mesh: trimesh.Trimesh,
    terrain_idx: int,
) -> TerrainGeometry:
    """Create a collision hfield from terrain mesh top-surface samples."""
    resolution = float(cfg.collision_hfield_resolution)
    if resolution <= 0.0:
        raise ValueError("collision_hfield_resolution must be greater than zero.")

    nrow = max(int(np.round(cfg.size[0] / resolution)) + 1, 2)
    ncol = max(int(np.round(cfg.size[1] / resolution)) + 1, 2)
    x = np.linspace(0.0, cfg.size[0], nrow, dtype=np.float64)
    y = np.linspace(0.0, cfg.size[1], ncol, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    sample_xy, sample_z = _top_surface_samples(
        terrain_mesh,
        cfg.collision_hfield_normal_z_threshold,
        resolution,
    )
    sample_xy = np.clip(sample_xy, a_min=(0.0, 0.0), a_max=(cfg.size[0], cfg.size[1]))

    height = interpolate.griddata(
        sample_xy,
        sample_z,
        (xx, yy),
        method="linear",
    )
    nan_mask = np.isnan(height)
    if np.any(nan_mask):
        height[nan_mask] = interpolate.griddata(
            sample_xy,
            sample_z,
            (xx[nan_mask], yy[nan_mask]),
            method="nearest",
        )
    height = np.nan_to_num(height, nan=0.0)

    elevation_min = float(np.min(height))
    elevation_max = float(np.max(height))
    elevation_range = max(elevation_max - elevation_min, 1.0e-6)
    normalized_height = (height - elevation_min) / elevation_range
    base_thickness = max(
        elevation_range * cfg.collision_hfield_base_thickness_ratio,
        1.0e-6,
    )

    hfield_name = f"motion_matched_hfield_t{terrain_idx}_{uuid.uuid4().hex}"
    hfield = spec.add_hfield(
        name=hfield_name,
        size=[
            cfg.size[0] / 2,
            cfg.size[1] / 2,
            elevation_range,
            base_thickness,
        ],
        nrow=nrow,
        ncol=ncol,
        userdata=normalized_height.astype(np.float32).reshape(-1).tolist(),
    )
    hfield_geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield_name,
        pos=(cfg.size[0] / 2, cfg.size[1] / 2, elevation_min - base_thickness),
    )
    hfield_geom.group = 1
    hfield_geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
    return TerrainGeometry(geom=hfield_geom, hfield=hfield)


def motion_matched_terrain(
    cfg: mesh_terrains_cfg.MotionMatchedTerrainCfg,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
) -> TerrainOutput:
    """Generate a motion-matched terrain based on the difficulty level and configuration.

    Args:
        difficulty (float): The difficulty level for the terrain.
        cfg (mesh_terrains_cfg.MotionMatchedTerrainCfg): Configuration for the motion-matched terrain.

    Returns:
        TerrainOutput: Spawn origin and generated MuJoCo terrain geometry.
    """
    del rng

    # Load the YAML file containing terrain and motion data
    with open(cfg.metadata_yaml) as file:
        data = yaml.safe_load(file)

    # Extract terrains and motions from the YAML data
    terrains = data["terrains"]  # order matters

    if cfg.collision_coacd:
        # Precompute missing CoACD caches once to better utilize multi-core CPU.
        _prewarm_coacd_disk_cache(cfg, terrains)

    terrain_idx = int(np.clip(difficulty * len(terrains), 0, len(terrains) - 1))
    selected_terrain = terrains[terrain_idx]

    terrain_file = selected_terrain["terrain_file"]
    terrain_abspath = os.path.join(cfg.path, terrain_file)
    terrain_size = (float(cfg.size[0]), float(cfg.size[1]))
    terrain_mesh, border_height = _load_motion_matched_terrain_mesh(
        terrain_abspath=terrain_abspath,
        terrain_file=terrain_file,
        size=terrain_size,
    )
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, -border_height])

    # Encode terrain index in mesh name so runtime can recover terrain-to-origin mapping
    # without relying on custom terrain importer compatibility fields.
    mesh_name = f"motion_matched_t{terrain_idx}_{uuid.uuid4().hex}"
    spec.add_mesh(
        name=mesh_name,
        uservert=np.asarray(terrain_mesh.vertices, dtype=np.float32).reshape(-1).tolist(),
        userface=np.asarray(terrain_mesh.faces, dtype=np.int32).reshape(-1).tolist(),
    )
    geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
        pos=(0.0, 0.0, 0.0),
    )
    geometries = [TerrainGeometry(geom=geom)]
    enabled_modes = sum([
        bool(cfg.face_box_collision),
        bool(cfg.collision_hfield),
        bool(cfg.collision_coacd),
    ])
    if enabled_modes > 1:
        raise ValueError(
            "face_box_collision, collision_hfield, and collision_coacd are mutually exclusive. "
            "Enable only one of them."
        )
    if cfg.collision_coacd:
        # Keep mesh for rendering/depth sensing but route physics contacts to CoACD convex hulls.
        # Root cause: MuJoCo's polyhedral mesh collision treats the interior of any closed
        # mesh as solid, reporting negative-dist contacts for objects near the inner surface.
        # CoACD decomposes the concave mesh into approximate convex hulls for correct collision.
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geometries.extend(
            _add_collision_coacd(cfg, spec, terrain_mesh, terrain_idx, terrain_abspath)
        )
    elif cfg.face_box_collision:
        # Keep mesh for rendering/depth sensing but route physics contacts to per-face boxes.
        # Root cause: MuJoCo's polyhedral mesh collision treats the interior of any closed
        # mesh as solid, reporting negative-dist contacts for objects near the inner surface.
        # Per-face box geoms placed on the outward side give correct surface-only collision.
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geometries.extend(
            _add_collision_face_boxes(cfg, spec, terrain_mesh)
        )
    elif cfg.collision_hfield:
        # Keep mesh for rendering/depth sensing but route physics contacts to hfield.
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = 2
        geometries.append(
            _add_collision_hfield_from_mesh(cfg, spec, terrain_mesh, terrain_idx)
        )
    return TerrainOutput(origin=origin, geometries=geometries)


@generate_wall
def floating_box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.PerlinMeshFloatingBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a floating box terrain."""

    # resolve the terrain configuration
    # height of the floating box above the ground
    if isinstance(cfg.floating_height, (tuple, list)):
        floating_height = cfg.floating_height[1] - difficulty * (cfg.floating_height[1] - cfg.floating_height[0])
    else:
        floating_height = cfg.floating_height

    # length of the floating box
    if isinstance(cfg.box_length, (tuple, list)):
        box_length = cfg.box_length[1] - difficulty * (cfg.box_length[1] - cfg.box_length[0])
    else:
        box_length = cfg.box_length

    # height of the floating box
    if isinstance(cfg.box_height, (tuple, list)):
        box_height = np.random.uniform(*cfg.box_height)
    else:
        box_height = cfg.box_height

    # width of the floating box
    if cfg.box_width is None:
        box_width = cfg.size[0]
    else:
        box_width = cfg.box_width

    # initialize the list of meshes
    meshes_list = list()

    # extract quantities
    total_height = floating_height + box_height
    # constants for terrain generation
    terrain_height = 0.0

    # generate the box mesh
    dim = (box_width, box_length, box_height)
    pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], floating_height + box_height / 2)
    box_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(box_mesh)

    # generate the ground

    if cfg.perlin_cfg is None:
        dim = (cfg.size[0], cfg.size[1], terrain_height)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        meshes_list.append(ground_mesh)
    else:
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        if cfg.no_perlin_at_obstacle is True:
            box_width_px = int(box_width / cfg.horizontal_scale)
            box_length_px = int(box_length / cfg.horizontal_scale)
            box_width_start_px = int((cfg.size[0] - box_width) / 2 / cfg.horizontal_scale)
            box_length_start_px = int((cfg.size[1] - box_length) / 2 / cfg.horizontal_scale)
            perlin_noise[
                box_width_start_px : box_width_start_px + box_width_px,
                box_length_start_px : box_length_start_px + box_length_px,
            ] = 0
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        meshes_list.append(ground_mesh)

    # specify the origin of the terrain
    origin = np.array([pos[0], pos[1], total_height])

    return meshes_list, origin


@generate_wall
def random_multi_box_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.PerlinMeshRandomMultiBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generates a terrain containing multiple boxes with random size and orientation."""

    box_height_range = cfg.box_height_range
    box_length_range = cfg.box_length_range
    box_width_range = cfg.box_width_range

    if isinstance(cfg.box_height_mean, (tuple, list)):
        if cfg.box_height_mean[0] < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")
        box_height_mean = cfg.box_height_mean[0] + difficulty * (cfg.box_height_mean[1] - cfg.box_height_mean[0])
    else:
        box_height_mean = cfg.box_height_mean
        if box_height_mean < box_height_range:
            raise RuntimeError("The minimum box height mean is smaller than the box height half range.")

    if isinstance(cfg.box_length_mean, (tuple, list)):
        if cfg.box_length_mean[0] < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")
        box_length_mean = cfg.box_length_mean[0] + difficulty * (cfg.box_length_mean[1] - cfg.box_length_mean[0])
    else:
        box_length_mean = cfg.box_length_mean
        if box_length_mean < box_length_range:
            raise RuntimeError("The minimum box length mean is smaller than the box length half range.")

    if isinstance(cfg.box_width_mean, (tuple, list)):
        if cfg.box_width_mean[0] < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")
        box_width_mean = cfg.box_width_mean[0] + difficulty * (cfg.box_width_mean[1] - cfg.box_width_mean[0])
    else:
        box_width_mean = cfg.box_width_mean
        if box_width_mean < box_width_range:
            raise RuntimeError("The minimum box width mean is smaller than the box width half range.")

    generation_ratio = cfg.generation_ratio

    width = cfg.size[0]
    length = cfg.size[1]

    mesh_list = []

    num_boxes = int(generation_ratio * (width * length) / (box_length_mean * box_width_mean))
    num_boxes = max(1, num_boxes)
    if cfg.perlin_cfg is None:
        dim = (cfg.size[0], cfg.size[1], 0.0)
        pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0)
        ground_mesh = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
        mesh_list.append(ground_mesh)
    else:
        clean_ground_height_field = np.zeros(
            (int(cfg.size[0] / cfg.horizontal_scale) + 1, int(cfg.size[1] / cfg.horizontal_scale) + 1), dtype=np.int16
        )
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(
            difficulty,
            perlin_cfg,  # type: ignore[arg-type]
        )
        h, w = perlin_noise.shape
        ground_h, ground_w = clean_ground_height_field.shape
        pad_h_left = max(0, (ground_h - h) // 2)
        pad_h_right = max(0, ground_h - h - pad_h_left)
        pad_w_left = max(0, (ground_w - w) // 2)
        pad_w_right = max(0, ground_w - w - pad_w_left)
        pad_width = ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right))
        perlin_noise = np.pad(perlin_noise, pad_width, mode="constant", constant_values=0)
        ground_height_field = clean_ground_height_field + perlin_noise
        # convert to trimesh
        vertices, triangles = convert_height_field_to_mesh(
            ground_height_field, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
        )
        ground_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh_list.append(ground_mesh)

    if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
        box_perlin_cfg = cfg.box_perlin_cfg
        box_perlin_cfg.horizontal_scale = (
            cfg.horizontal_scale if box_perlin_cfg.horizontal_scale is None else box_perlin_cfg.horizontal_scale
        )
        box_perlin_cfg.vertical_scale = (
            cfg.vertical_scale if box_perlin_cfg.vertical_scale is None else box_perlin_cfg.vertical_scale
        )
        box_perlin_cfg.slope_threshold = (
            cfg.slope_threshold if box_perlin_cfg.slope_threshold is None else box_perlin_cfg.slope_threshold
        )

    platform_width = cfg.platform_width

    for i in range(num_boxes):
        box_width = box_width_mean + np.random.uniform(-1, 1) * box_width_range
        box_length = box_length_mean + np.random.uniform(-1, 1) * box_length_range
        box_height = box_height_mean + np.random.uniform(-1, 1) * box_height_range
        dim = (box_width, box_length, box_height)
        x = np.random.uniform(box_width / 2, width - box_width / 2)
        y = np.random.uniform(box_length / 2, length - box_length / 2)
        if (
            x > width / 2 - platform_width / 2 - box_width / 2 and x < width / 2 + platform_width / 2 + box_width / 2
        ) and (
            y > length / 2 - platform_width / 2 - box_length / 2
            and y < length / 2 + platform_width / 2 + box_length / 2
        ):
            continue
        pos = (x, y, box_height / 2)
        theta = np.random.uniform(0, 2 * np.pi)
        translation_matrix = trimesh.transformations.translation_matrix(pos)
        rotation_matrix = trimesh.transformations.rotation_matrix(theta, (0, 0, 1))
        transform = translation_matrix @ rotation_matrix
        box_mesh = trimesh.creation.box(extents=dim)
        # top_z=box_mesh.vertices[:, 2].max()
        # top_face_mask=np.all(box_mesh.vertices[box_mesh.faces][:,:,2] == top_z, axis=1)
        # box_mesh.update_faces(~top_face_mask)
        # box_mesh.remove_unreferenced_vertices()
        box_mesh.apply_transform(transform)
        mesh_list.append(box_mesh)
        if cfg.box_perlin_cfg is not None and cfg.no_perlin_at_obstacle is False:
            box_perlin_cfg.size = (box_width, box_length)
            perlin_noise = generate_perlin_noise(
                difficulty,
                box_perlin_cfg,  # type: ignore[arg-type]
            )
            vertices, triangles = convert_height_field_to_mesh(
                perlin_noise,
                box_perlin_cfg.horizontal_scale,
                box_perlin_cfg.vertical_scale,
                box_perlin_cfg.slope_threshold,
            )
            box_noise = trimesh.Trimesh(vertices=vertices, faces=triangles)
            center_offset = (-box_width / 2, -box_length / 2, 0)
            center_translation = trimesh.transformations.translation_matrix(center_offset)
            box_noise.apply_transform(center_translation)
            noise_pos = (x, y, box_height)
            translation_matrix = trimesh.transformations.translation_matrix(noise_pos)
            transform = translation_matrix @ rotation_matrix
            box_noise.apply_transform(transform)
            mesh_list.append(box_noise)

    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return mesh_list, origin
