# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-RClause

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
from isaaclab.managers import CommandTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from . import mdp


##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from ....assets.so101 import SO101_CFG, SO101_IK_JOINTS, SO101_GRIPPER_JOINTS, SO101_END_EFFECTOR
from ....assets.cube import CUBE_CFG

##
# Scene definition
##

TABLE_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/PackingTable/packing_table.usd"
ARM_URDF = "/opt/IsaacSim/custom_projects/so-101/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"


@configclass
class So101LiftSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/jaw",
                name="ee_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0))
            )
        ]
    )

    # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )

    # robot
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    # https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/PackingTable/packing_table.usd


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    arm_action = DifferentialInverseKinematicsActionCfg(asset_name="robot", joint_names=SO101_IK_JOINTS,
                                                        body_name=SO101_END_EFFECTOR,
                                                        controller=DifferentialIKControllerCfg(
                                                            command_type="pose",
                                                            use_relative_mode=True,
                                                            ik_method="dls"
                                                        ))
    gripper_action = JointPositionActionCfg(asset_name="robot", joint_names=SO101_GRIPPER_JOINTS, scale=0.5)


# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         # observation terms (order preserved)
#         cube_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")})

#         def __post_init__(self) -> None:
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # reset joints
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=SO101_GRIPPER_JOINTS + SO101_IK_JOINTS),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube")
        },
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="jaw",  # or whichever link/frame you're tracking
        resampling_time_range=(5.0, 5.0),  # how often to resample a new goal, in seconds
        # debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.05),
            pos_y=(-0.25, -0.2),
            pos_z=(0.2, 0.3),  # your target lift height range
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube")}
    )

    # sparse-ish: reward for lifting the cube above a minimal height threshold
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={"minimal_height": 1.10, "object_cfg": SceneEntityCfg("cube")},
    )

    # dense: once lifted, reward tracking toward the target height/position
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 1.10, "command_name": "object_pose"},
    )

    # penalty: discourage large/jerky actions
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-4,
    )

    # penalty: discourage excessive joint velocity
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass

    # # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # # (2) Cart out of bounds
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.80, "asset_cfg": SceneEntityCfg("cube")},
    )


##
# Environment configuration
##


@configclass
class So101LiftEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: So101LiftSceneCfg = So101LiftSceneCfg(num_envs=1024, env_spacing=4.0)  # 1024 envs
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 7
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_collision_stack_size = 1_200_000_000  # ~1.2GB, headroom above the 864MB minimum

