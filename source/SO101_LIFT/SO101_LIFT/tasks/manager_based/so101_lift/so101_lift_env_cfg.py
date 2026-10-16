# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 cube lifting, following the squint ``SO101LiftCube-v1`` task (squint/envs/lift.py).

Differences to squint:
- the arm is controlled with differential IK (relative end-effector pose) instead of joint-target deltas,
- the policy observes the cube through ground-truth state instead of the 16x16 wrist camera image,
- physics engine is Isaac Lab / PhysX instead of ManiSkill, so the grasp/table checks are re-implemented
  with a contact sensor (see mdp/rewards.py).
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from . import mdp


##
# Pre-defined configs
##

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from ....assets.so101 import (
    SO101_CFG, SO101_IK_JOINTS, SO101_GRIPPER_JOINTS, SO101_FIXED_JAW, SO101_FINGER_BODIES
)
from ....assets.cube import CUBE_CFG

# squint runs physics at 100 Hz and the policy at 10 Hz, with 50 steps (5 s) per episode
SIM_DT = 1 / 100
DECIMATION = 10
STEP_DT = SIM_DT * DECIMATION
EPISODE_LENGTH_S = 5.0

# squint trains on ``normalized_dense`` reward = dense reward / 3. Isaac Lab multiplies every reward term
# by ``weight * step_dt``, so the weights are divided by step_dt to get exactly squint's per-step reward.
REWARD_SCALE = 1.0 / (3.0 * STEP_DT)

# all joints, gripper last (same layout as squint's action / target_qpos)
SO101_JOINTS = SO101_IK_JOINTS + SO101_GRIPPER_JOINTS


##
# Scene definition
##

TABLE_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/PackingTable/packing_table.usd"
ARM_URDF = "/opt/IsaacSim/custom_projects/so-101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"


@configclass
class So101LiftSceneCfg(InteractiveSceneCfg):
    """Configuration for the SO-101 lift scene."""

    # tool center point (between the fingers)
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/jaw",
                name="ee_frame",
                offset=OffsetCfg(pos=(-0.02, -0.045, 0.0))
            )
        ]
    )

    # robot
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact forces on the robot links; the per-cube forces (force_matrix_w) are used for the grasp check
    robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        update_period=0.0,
        history_length=0,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=TABLE_URL)
    )

    cube: RigidObjectCfg = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube")


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Relative end-effector pose (dx, dy, dz, rotation vector) solved with differential IK. The squint policy
    # outputs actions in [-1, 1] (tanh), so the scale is the largest pose change per 10 Hz step:
    # 3 cm translation and 0.1 rad rotation (comparable to squint's 0.1 rad joint deltas).
    # The IK controls the fixed jaw (rigidly attached to the wrist), so opening/closing the gripper does not
    # move the IK target and the arm does not try to compensate for the gripper motion.
    arm_action = DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=SO101_IK_JOINTS,
                                                        body_name=SO101_FIXED_JAW,
                                                        scale=(0.03, 0.03, 0.03, 0.1, 0.1, 0.1),
                                                        controller=DifferentialIKControllerCfg(
                                                            command_type="pose",
                                                            use_relative_mode=True,
                                                            ik_method="dls"
                                                        ))
    gripper_action = JointPositionActionCfg(asset_name="robot", joint_names=SO101_GRIPPER_JOINTS, scale=0.5)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # squint agent state: noisy joint positions (5 deg std noise for sim2real) + controller joint target
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SO101_JOINTS, preserve_order=True)},
            noise=Gnoise(std=math.radians(5)),
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SO101_JOINTS, preserve_order=True)},
        )
        # ground-truth cube state (replaces squint's wrist camera image)
        cube_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        cube_quat = ObsTerm(func=mdp.object_quat_in_robot_root_frame)
        tcp_to_cube = ObsTerm(func=mdp.tcp_to_object_in_robot_root_frame)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events (squint's _initialize_episode + domain randomisation)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # robot starts at the rest pose + N(0, 0.02) joint noise
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_gaussian,
        mode="reset",
        params={"std": 0.02, "asset_cfg": SceneEntityCfg("robot", joint_names=SO101_JOINTS)},
    )

    # cube in a 0.1 x 0.1 m square around its default position with a random yaw
    # (squint uses a 0.2 x 0.2 m square, which does not fit the table well)
    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    # cube friction 0.1 - 0.5, no restitution
    cube_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "static_friction_range": (0.1, 0.5),
            "dynamic_friction_range": (0.1, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # high-friction gripper pads (squint's so101 gripper material: friction 2.0)
    finger_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=SO101_FINGER_BODIES),
            "static_friction_range": (2.0, 2.0),
            "dynamic_friction_range": (2.0, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # gripper speed randomisation per episode (squint: stiffness 500-2000, damping 50-200, i.e. 0.4x-1.6x)
    gripper_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=SO101_GRIPPER_JOINTS),
            "stiffness_distribution_params": (0.4, 1.6),
            "damping_distribution_params": (0.4, 1.6),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    """squint LiftCube dense reward, normalised by 3 (see mdp/rewards.py)."""

    reaching_cube = RewTerm(func=mdp.reaching_cube, weight=1.0 * REWARD_SCALE)
    cube_grasped = RewTerm(func=mdp.cube_grasped, weight=1.0 * REWARD_SCALE)
    return_to_rest = RewTerm(func=mdp.return_to_rest_while_grasped, weight=1.0 * REWARD_SCALE)
    touching_table = RewTerm(func=mdp.fingers_touching_table, weight=-3.0 * REWARD_SCALE)
    cube_not_lifted = RewTerm(func=mdp.cube_not_lifted, weight=-1.0 * REWARD_SCALE)

    # weight 0: only logs Metrics/success_once and Metrics/success_at_end
    success_metrics = RewTerm(func=mdp.lift_success_metrics, weight=0.0)


@configclass
class TerminationsCfg:
    """squint trains with ignore_terminations=True: episodes only end on the time limit."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class So101LiftEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: So101LiftSceneCfg = So101LiftSceneCfg(num_envs=1024, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = DECIMATION
        self.episode_length_s = EPISODE_LENGTH_S
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_collision_stack_size = 1_200_000_000  # ~1.2GB, headroom above the 864MB minimum
