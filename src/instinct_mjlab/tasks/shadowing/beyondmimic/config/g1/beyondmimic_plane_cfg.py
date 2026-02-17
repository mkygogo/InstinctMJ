from __future__ import annotations

from copy import deepcopy

import yaml

import mjlab.envs.mdp as mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg
from mjlab.managers import EventTermCfg
from mjlab.managers import ObservationGroupCfg
from mjlab.managers import ObservationTermCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

import instinct_mjlab.envs.mdp as instinct_mdp
from instinct_mjlab.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    G1_MJCF_PATH,
    beyondmimic_action_scale,
)
from instinct_mjlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinct_mjlab.monitors import (
    ActuatorMonitorTerm,
    MonitorTermCfg,
    MotionReferenceMonitorTerm,
    RewardSumMonitorTerm,
    ShadowingJointPosMonitorTerm,
    ShadowingJointReferenceMonitorTerm,
    ShadowingJointVelMonitorTerm,
    ShadowingLinkPosMonitorTerm,
    ShadowingPositionMonitorTerm,
    ShadowingRotationMonitorTerm,
)
from instinct_mjlab.motion_reference import MotionReferenceManagerCfg
from instinct_mjlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinct_mjlab.motion_reference.utils import motion_interpolate_bilinear
from instinct_mjlab.utils.datasets import resolve_datasets_root

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG
_DATASETS_ROOT = resolve_datasets_root()
_CONTACT_SENSOR_NAME = "contact_forces"

# Motion configuration
MOTION_NAME = "LafanKungfu1"
_hacked_selected_file_ = "fightAndSports1_subject1_retargetted.npz"
MOTION_NAME = "LafanSprint1"
_hacked_selected_file_ = "sprint1_subject2_retargetted.npz"

with open(f"/tmp/{MOTION_NAME}.yaml", "w") as f:
    yaml.dump(
        {
            "selected_files": [
                _hacked_selected_file_,
            ],
        },
        f,
    )


def _make_amass_motion_cfg() -> AmassMotionCfgBase:
    """AMASS motion configuration for BeyondMimic."""

    return AmassMotionCfgBase(
        path=str(_DATASETS_ROOT / "UbisoftLAFAN1_GMR_g1_29dof_torsoBase_retargetted_instinctnpz"),
        retargetting_func=None,
        filtered_motion_selection_filepath=f"/tmp/{MOTION_NAME}.yaml",
        motion_start_from_middle_range=[0.0, 0.8],
        motion_start_height_offset=0.0,
        ensure_link_below_zero_ground=False,
        buffer_device="output_device",
        motion_interpolate_func=motion_interpolate_bilinear,
        velocity_estimation_method="frontbackward",
        motion_bin_length_s=1.0,
    )


def _make_motion_reference_cfg(*, debug_vis: bool) -> MotionReferenceManagerCfg:
    return MotionReferenceManagerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        robot_model_path=G1_MJCF_PATH,
        reference_prim_path="/World/envs/env_.*/RobotReference/torso_link",
        link_of_interests=[
            "pelvis",
            "torso_link",
            "left_shoulder_roll_link",
            "right_shoulder_roll_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_hip_roll_link",
            "right_hip_roll_link",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ],
        symmetric_augmentation_link_mapping=None,
        symmetric_augmentation_joint_mapping=None,
        symmetric_augmentation_joint_reverse_buf=None,
        frame_interval_s=0.0,
        update_period=0.02,
        num_frames=1,
        visualizing_robot_offset=(0.0, 1.5, 0.0),
        visualizing_robot_from="reference_frame",
        motion_buffers={
            MOTION_NAME: _make_amass_motion_cfg(),
        },
        mp_split_method="Even",
        debug_vis=debug_vis,
    )


def _make_scene_cfg(*, play: bool, motion_reference_cfg: MotionReferenceManagerCfg) -> SceneCfg:
    contact_forces = ContactSensorCfg(
        name=_CONTACT_SENSOR_NAME,
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="maxforce",
        history_length=3,
        track_air_time=True,
    )

    entities = {
        "robot": deepcopy(G1_CFG),
    }
    if play and motion_reference_cfg.debug_vis:
        entities["robot_reference"] = deepcopy(G1_CFG)

    return SceneCfg(
        num_envs=1 if play else 4096,
        env_spacing=2.5 if play else 4.0,
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities=entities,
        sensors=(
            contact_forces,
            motion_reference_cfg,
        ),
    )


def _actions_cfg() -> dict[str, mdp.JointPositionActionCfg]:
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=deepcopy(beyondmimic_action_scale),
        ),
    }


def _commands_cfg() -> dict[str, object]:
    return {
        "position_ref_command": instinct_mdp.PositionRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            anchor_frame="robot",
        ),
        "rotation_ref_command": instinct_mdp.RotationRefCommandCfg(
            realtime_mode=True,
            current_state_command=False,
            in_base_frame=True,
            rotation_mode="tannorm",
        ),
        "joint_pos_ref_command": instinct_mdp.JointPosRefCommandCfg(current_state_command=False),
        "joint_vel_ref_command": instinct_mdp.JointVelRefCommandCfg(current_state_command=False),
    }


def _observations_cfg(link_of_interests: list[str]) -> dict[str, ObservationGroupCfg]:
    return {
        "actor": ObservationGroupCfg(
            terms={
                # BeyondMimic specific reference observations
                "joint_pos_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_pos_ref_command"},
                ),
                "joint_vel_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_vel_ref_command"},
                ),
                "position_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "position_ref_command"},
                    noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
                ),
                "rotation_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "rotation_ref_command"},
                    noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
                ),
                # proprioception
                "base_lin_vel": ObservationTermCfg(
                    func=mdp.base_lin_vel,
                    noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
                ),
                "base_ang_vel": ObservationTermCfg(
                    func=mdp.base_ang_vel,
                    noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
                ),
                "joint_pos": ObservationTermCfg(
                    func=mdp.joint_pos_rel,
                    noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
                ),
                "joint_vel": ObservationTermCfg(
                    func=mdp.joint_vel_rel,
                    noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
                ),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            },
            enable_corruption=True,
            concatenate_terms=False,
        ),
        "critic": ObservationGroupCfg(
            terms={
                # BeyondMimic specific reference observations
                "joint_pos_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_pos_ref_command"},
                ),
                "joint_vel_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "joint_vel_ref_command"},
                ),
                "position_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "position_ref_command"},
                ),
                "rotation_ref": ObservationTermCfg(
                    func=mdp.generated_commands,
                    params={"command_name": "rotation_ref_command"},
                ),
                # proprioception
                "link_pos": ObservationTermCfg(
                    func=instinct_mdp.link_pos_b,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            name="robot",
                            body_names=link_of_interests,
                            preserve_order=True,
                        ),
                    },
                ),
                "link_rot": ObservationTermCfg(
                    func=instinct_mdp.link_tannorm_b,
                    params={
                        "asset_cfg": SceneEntityCfg(
                            name="robot",
                            body_names=link_of_interests,
                            preserve_order=True,
                        ),
                    },
                ),
                "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            },
            enable_corruption=False,
            concatenate_terms=False,
        ),
    }


def _rewards_cfg() -> dict[str, RewardTermCfg | None]:
    return {
        "motion_global_root_pos": RewardTermCfg(
            func=instinct_mdp.base_position_tracking_gauss,
            weight=0.5,
            params={
                "tracking_sigma": 0.3,
                "tracking_torlerance": 0.0,
            },
        ),
        "motion_global_root_ori": RewardTermCfg(
            func=instinct_mdp.base_rot_tracking_gauss,
            weight=0.5,
            params={
                "tracking_sigma": 0.4,
                "tracking_torlerance": 0.0,
            },
        ),
        "motion_body_pos": RewardTermCfg(
            func=instinct_mdp.link_pos_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.3,
            },
        ),
        "motion_body_ori": RewardTermCfg(
            func=instinct_mdp.link_rot_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "in_base_frame": False,
                "in_relative_world_frame": True,
                "std": 0.4,
            },
        ),
        "motion_body_lin_vel": RewardTermCfg(
            func=instinct_mdp.link_lin_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 1.0,
            },
        ),
        "motion_body_ang_vel": RewardTermCfg(
            func=instinct_mdp.link_ang_vel_imitation_gauss,
            weight=1.0,
            params={
                "combine_method": "mean_prod",
                "std": 3.14,
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_limit": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        ),
        "undesired_contacts": RewardTermCfg(
            func=instinct_mdp.undesired_contacts,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg(
                    _CONTACT_SENSOR_NAME,
                    body_names=[
                        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                    ],
                ),
                "threshold": 1.0,
            },
        ),
    }


def _events_cfg() -> dict[str, EventTermCfg | None]:
    return {
        # startup
        "physics_material": EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.2),
                "restitution_range": (0.0, 0.5),
                "num_buckets": 64,
            },
        ),
        "add_joint_default_pos": EventTermCfg(
            func=instinct_mdp.randomize_default_joint_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "offset_distribution_params": (-0.01, 0.01),
                "operation": "add",
                "distribution": "uniform",
            },
        ),
        "base_com": EventTermCfg(
            func=instinct_mdp.randomize_rigid_body_coms,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "coms_x_distribution_params": (-0.025, 0.025),
                "coms_y_distribution_params": (-0.05, 0.05),
                "coms_z_distribution_params": (-0.05, 0.05),
                "distribution": "uniform",
            },
        ),
        # interval
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        # for motion initialization and reset
        "match_motion_ref_with_scene": EventTermCfg(
            func=instinct_mdp.match_motion_ref_with_scene,
            mode="startup",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            },
        ),
        "reset_robot": EventTermCfg(
            func=instinct_mdp.reset_robot_state_by_reference,
            mode="reset",
            params={
                "motion_ref_cfg": SceneEntityCfg("motion_reference"),
                "asset_cfg": SceneEntityCfg("robot"),
                "position_offset": [0.0, 0.0, 0.0],
                "dof_vel_ratio": 1.0,
                "base_lin_vel_ratio": 1.0,
                "base_ang_vel_ratio": 1.0,
                # Pose randomization (+-5cm position, +-6degrees rotation)
                "randomize_pose_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.01, 0.01),
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.2, 0.2),
                },
                # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
                "randomize_velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.2, 0.2),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
                # Joint position randomization (+-0.1 rad)
                "randomize_joint_pos_range": (-0.1, 0.1),
            },
        ),
        "bin_fail_counter_smoothing": EventTermCfg(
            func=instinct_mdp.beyondmimic_bin_fail_counter_smoothing,
            mode="interval",
            interval_range_s=(0.02, 0.02),  # every environment step
            params={
                "curriculum_name": "beyond_adaptive_sampling",
            },
        ),
    }


def _curriculum_cfg() -> dict[str, CurriculumTermCfg | None]:
    return {
        "beyond_adaptive_sampling": CurriculumTermCfg(  # type: ignore[arg-type]
            func=instinct_mdp.BeyondMimicAdaptiveWeighting,
        ),
    }


def _terminations_cfg() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "base_pos_too_far": TerminationTermCfg(
            func=instinct_mdp.pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "distance_threshold": 0.25,
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
                "height_only": True,
            },
        ),
        "base_pg_too_far": TerminationTermCfg(
            func=instinct_mdp.projected_gravity_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "projected_gravity_threshold": 0.8,  # distance on z-axis of projected gravity
                "check_at_keyframe_threshold": -1,
                "print_reason": False,
            },
        ),
        "link_pos_too_far": TerminationTermCfg(
            func=instinct_mdp.link_pos_far_from_ref,
            time_out=False,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "reference_cfg": SceneEntityCfg(
                    "motion_reference",
                    body_names=[
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    preserve_order=True,
                ),
                "distance_threshold": 0.25,
                "in_base_frame": False,
                "check_at_keyframe_threshold": -1,
                "height_only": True,
                "print_reason": False,
            },
        ),
        "dataset_exhausted": TerminationTermCfg(
            func=instinct_mdp.dataset_exhausted,
            time_out=True,
            params={
                "reference_cfg": SceneEntityCfg("motion_reference"),
                "print_reason": False,
            },
        ),
        "out_of_border": TerminationTermCfg(
            func=instinct_mdp.terrain_out_of_bounds,
            time_out=True,
            params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
        ),
    }


def _monitors_cfg(*, play: bool) -> dict[str, MonitorTermCfg]:
    monitors = {
        "dataset": MonitorTermCfg(
            func=MotionReferenceMonitorTerm,
            params=dict(
                asset_cfg=SceneEntityCfg("motion_reference"),
                sample_stat_interval=500,
                top_n_samples=5,
            ),
        ),
        "shadowing_position": MonitorTermCfg(
            func=ShadowingPositionMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                check_at_keyframe_threshold=0.03,
            ),
        ),
        "shadowing_rotation": MonitorTermCfg(
            func=ShadowingRotationMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),
        "shadowing_joint_pos": MonitorTermCfg(
            func=ShadowingJointPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),
        "shadowing_joint_vel": MonitorTermCfg(
            func=ShadowingJointVelMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                masking=True,
            ),
        ),
        "shadowing_link_pos": MonitorTermCfg(
            func=ShadowingLinkPosMonitorTerm,
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
                in_base_frame=True,
                masking=True,
            ),
        ),
    }
    if not play:
        return monitors

    # add PLAY-specific monitor term
    monitors["shoulder_actuator"] = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_shoulder_roll.*"),
            "torque_plot_scale": 1e-2,
            "joint_power_plot_scale": 1e-1,
        },
    )
    monitors["waist_actuator"] = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="waist_roll.*"),
            "torque_plot_scale": 1e-2,
            "joint_power_plot_scale": 1e-1,
        },
    )
    monitors["knee_actuator"] = MonitorTermCfg(
        func=ActuatorMonitorTerm,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", joint_names="left_knee.*"),
            "torque_plot_scale": 1e-2,
            "joint_power_plot_scale": 1e-1,
        },
    )
    monitors["reward_sum"] = MonitorTermCfg(func=RewardSumMonitorTerm)
    monitors["reference_stat_case"] = MonitorTermCfg(
        func=ShadowingJointReferenceMonitorTerm,
        params=dict(
            reference_cfg=SceneEntityCfg(
                "motion_reference",
                joint_names=[
                    "left_hip_pitch.*",
                ],
            ),
        ),
    )
    return monitors


def _viewer_cfg(play: bool) -> ViewerConfig:
    if not play:
        return ViewerConfig()
    return ViewerConfig(
        lookat=(0.0, 0.75, 0.0),
        distance=4.1231,
        elevation=14.0362,
        azimuth=0.0,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
    )


def _apply_play_overrides(cfg: InstinctLabRLEnvCfg, motion_reference_cfg: MotionReferenceManagerCfg) -> None:
    # spawn the robot randomly in the grid (instead of their terrain levels)
    cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 3
        cfg.scene.terrain.terrain_generator.num_cols = 3
        cfg.scene.terrain.terrain_generator.curriculum = False

    motion_reference_cfg.symmetric_augmentation_joint_mapping = None
    motion_reference_cfg.visualizing_marker_types = ["relative_links"]
    cfg.curriculum["beyond_adaptive_sampling"] = None
    cfg.events["push_robot"] = None
    cfg.events["bin_fail_counter_smoothing"] = None

    # If you want to play the motion from start and till the end, uncomment the following lines
    cfg.episode_length_s = 6000.0
    motion_buffer = motion_reference_cfg.motion_buffers[MOTION_NAME]
    motion_buffer.motion_start_from_middle_range = [0.0, 0.0]
    motion_buffer.motion_bin_length_s = None

    # enable print_reason option in the termination terms
    for term in cfg.terminations.values():
        if term is None:
            continue
        if "print_reason" in term.params:
            term.params["print_reason"] = True

    # enable debug_vis option in commands
    for cmd in cfg.commands.values():
        if hasattr(cmd, "debug_vis"):
            cmd.debug_vis = True


def _build_run_name(cfg: InstinctLabRLEnvCfg) -> str:
    actor_terms = cfg.observations["actor"].terms
    return "".join(
        [
            "G1BeyondMimic",
            (
                "_linVelObs"
                if actor_terms.get("base_lin_vel", None) is not None
                and actor_terms["base_lin_vel"].scale != 0.0
                else ""
            ),
            f"_{MOTION_NAME}",
            ("_noPush" if cfg.events.get("push_robot", None) is None else ""),
            ("_noContactPenalty" if cfg.rewards.get("undesired_contacts", None) is None else ""),
            "_GmrMotion",
        ]
    )


def g1_beyondmimic_plane_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
    motion_reference_cfg = _make_motion_reference_cfg(debug_vis=play)
    scene = _make_scene_cfg(play=play, motion_reference_cfg=motion_reference_cfg)

    cfg = InstinctLabRLEnvCfg(
        scene=scene,
        actions=_actions_cfg(),
        observations=_observations_cfg(link_of_interests=list(motion_reference_cfg.link_of_interests)),
        commands=_commands_cfg(),
        rewards=_rewards_cfg(),
        events=_events_cfg(),
        curriculum=_curriculum_cfg(),
        terminations=_terminations_cfg(),
        monitors=_monitors_cfg(play=play),
        viewer=_viewer_cfg(play),
        decimation=4,
        episode_length_s=10.0,
    )

    cfg.sim.mujoco.timestep = 1.0 / 50.0 / cfg.decimation

    assert (
        len(list(motion_reference_cfg.motion_buffers.keys())) == 1
    ), "Only support single motion buffer for now"

    if play:
        _apply_play_overrides(cfg, motion_reference_cfg)

    cfg.run_name = _build_run_name(cfg)
    return cfg


def G1BeyondMimicPlaneEnvCfg() -> ManagerBasedRlEnvCfg:
    """Compatibility callable that returns the train env config."""

    return g1_beyondmimic_plane_env_cfg(play=False)


def G1BeyondMimicPlaneEnvCfg_PLAY() -> ManagerBasedRlEnvCfg:
    """Compatibility callable that returns the play env config."""

    return g1_beyondmimic_plane_env_cfg(play=True)


__all__ = [
    "G1BeyondMimicPlaneEnvCfg",
    "G1BeyondMimicPlaneEnvCfg_PLAY",
    "g1_beyondmimic_plane_env_cfg",
]
