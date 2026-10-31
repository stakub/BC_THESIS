"""Rewards of the squint ``SO101LiftCube-v1`` task (squint/envs/lift.py), ported to Isaac Lab.

Squint's dense reward (before normalisation) is::

    reward  = 1 - tanh(5 * |cube - tcp|)                         # reaching
    reward += is_grasped                                         # grasping
    reward += exp(-2 * |target_qpos[:5] - rest_qpos[:5]|) * is_grasped   # bring arm back to rest pose
    reward -= 3 * fingers_touching_table
    reward -= 1 * (not cube_lifted)                              # lift the cube fast

and the normalised reward used for training is ``reward / 3``. Each line is one reward term here; the
``1 / 3`` normalisation (and the ``1 / step_dt`` that cancels Isaac Lab's ``weight * dt`` scaling) is applied
through the term weights in the env config.

All terms read the same per-step quantities from :func:`lift_info`, which mirrors squint's ``evaluate()``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from SO101_LIFT.assets.so101 import SO101_FINGER_BODIES, SO101_IK_JOINTS

from .observations import tcp_pos_w

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

# names of the scene entities / manager terms used by the squint lift task
ROBOT_NAME = "robot"
CUBE_NAME = "cube"
EE_FRAME_NAME = "ee_frame"
CONTACT_SENSOR_NAME = "robot_contacts"

# thresholds, taken from squint (envs/lift.py and envs/robot/so101.py)
GRASP_MIN_FORCE = 0.5  # N, per finger
GRASP_MIN_OPPOSING_ANGLE = 90.0  # deg, see note in lift_info()
TOUCH_MIN_FORCE = 1e-2  # N
REACHED_OBJECT_DIST = 0.03  # m
REST_QPOS_THRESHOLD = 0.2  # rad (L2 over the 5 arm joints)
LIFT_THRESHOLD = 1e-3  # m above the resting height


def _finger_body_ids(env: ManagerBasedRLEnv, sensor: ContactSensor) -> list[int]:
    ids = getattr(env, "_squint_finger_body_ids", None)
    if ids is None:
        names = sensor.body_names
        missing = [n for n in SO101_FINGER_BODIES if n not in names]
        if missing:
            raise ValueError(
                f"Finger bodies {missing} not found in the robot contact sensor. Available bodies: {names}. "
                "Update SO101_FIXED_JAW / SO101_END_EFFECTOR in SO101_LIFT/assets/so101.py."
            )
        ids = [names.index(n) for n in SO101_FINGER_BODIES]
        env._squint_finger_body_ids = ids
    return ids


def _arm_joint_ids(env: ManagerBasedRLEnv, robot: Articulation) -> list[int]:
    ids = getattr(env, "_squint_arm_joint_ids", None)
    if ids is None:
        ids, _ = robot.find_joints(SO101_IK_JOINTS, preserve_order=True)
        env._squint_arm_joint_ids = ids
    return ids


def _episode_buffers(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    bufs = getattr(env, "_squint_episode_buffers", None)
    if bufs is None:
        bufs = {
            # lowest cube height seen this episode == resting height on the table
            "cube_rest_z": torch.full((env.num_envs,), float("inf"), device=env.device),
            "success": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "success_once": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        env._squint_episode_buffers = bufs
    return bufs


def lift_info(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Squint's ``evaluate()``: task flags/metrics for the current step (computed once per env step)."""
    cache = getattr(env, "_squint_lift_info", None)
    if cache is not None and cache["step"] == env.common_step_counter:
        return cache

    robot: Articulation = env.scene[ROBOT_NAME]
    cube: RigidObject = env.scene[CUBE_NAME]
    sensor: ContactSensor = env.scene[CONTACT_SENSOR_NAME]
    bufs = _episode_buffers(env)

    cube_pos = cube.data.root_pos_w
    tcp_pos = tcp_pos_w(env, SceneEntityCfg(EE_FRAME_NAME))
    # NaN-safe: an env whose physics blew up gets "far away" distances for the one step before the
    # invalid_state termination resets it (comparisons with NaN already give False for the flags)
    tcp_to_cube_dist = torch.nan_to_num(torch.linalg.norm(cube_pos - tcp_pos, dim=-1), nan=10.0, posinf=10.0)

    # --- contacts of the two fingers ---
    finger_ids = _finger_body_ids(env, sensor)
    cube_forces = sensor.data.force_matrix_w[:, finger_ids, 0, :]  # (N, 2, 3) force from the cube on each finger
    net_forces = sensor.data.net_forces_w[:, finger_ids, :]  # (N, 2, 3) force from everything
    cube_force_norm = torch.linalg.norm(cube_forces, dim=-1)

    # Squint: both fingers push on the item with >= 0.5 N, and each contact force points along the finger's
    # opening direction (within 110 deg). The finger frame axes of this USD are unknown, so we use the
    # frame-free equivalent: both forces >= 0.5 N and roughly opposing each other (the item is squeezed).
    f_fixed, f_moving = cube_forces[:, 0], cube_forces[:, 1]
    cos_angle = torch.nn.functional.cosine_similarity(f_fixed, f_moving, dim=-1)
    opposing = cos_angle <= math.cos(math.radians(GRASP_MIN_OPPOSING_ANGLE))
    is_grasped = (cube_force_norm >= GRASP_MIN_FORCE).all(dim=-1) & opposing

    # Squint: is_touching(table) on the finger links. The only other thing the fingers can touch is the
    # table, so "contact force not coming from the cube" == touching the table.
    other_force = torch.linalg.norm(net_forces - cube_forces, dim=-1)
    robot_touching_table = (other_force >= TOUCH_MIN_FORCE).any(dim=-1)

    # --- distance of the joint *target* (here: the IK solution) to the rest pose, arm joints only ---
    arm_ids = _arm_joint_ids(env, robot)
    rest_qpos = robot.data.default_joint_pos[:, arm_ids]
    distance_to_rest_qpos = torch.nan_to_num(
        torch.linalg.norm(robot.data.joint_pos_target[:, arm_ids] - rest_qpos, dim=-1), nan=10.0, posinf=10.0
    )
    reached_rest_qpos = distance_to_rest_qpos < REST_QPOS_THRESHOLD

    # --- lifted: squint uses z >= half_size + 1e-3 (table at z=0). We do not hard-code the table height,
    # the resting height is the lowest cube height seen during the episode instead.
    cube_z = cube_pos[:, 2]
    bufs["cube_rest_z"] = torch.minimum(bufs["cube_rest_z"], cube_z)
    item_lifted = cube_z >= bufs["cube_rest_z"] + LIFT_THRESHOLD

    success = item_lifted & is_grasped & reached_rest_qpos
    bufs["success"] = success
    bufs["success_once"] |= success

    cache = {
        "step": env.common_step_counter,
        "tcp_to_cube_dist": tcp_to_cube_dist,
        "reached_object": tcp_to_cube_dist < REACHED_OBJECT_DIST,
        "is_item_grasped": is_grasped,
        "distance_to_rest_qpos": distance_to_rest_qpos,
        "robot_touching_table": robot_touching_table,
        "item_lifted": item_lifted,
        "success": success,
    }
    env._squint_lift_info = cache
    return cache


"""
Reward terms.
"""


def reaching_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``1 - tanh(5 * |cube - tcp|)``."""
    return 1.0 - torch.tanh(5.0 * lift_info(env)["tcp_to_cube_dist"])


def cube_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 if the cube is grasped."""
    return lift_info(env)["is_item_grasped"].float()


def return_to_rest_while_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``exp(-2 * |target_qpos - rest_qpos|)`` while the cube is grasped."""
    info = lift_info(env)
    return torch.exp(-2.0 * info["distance_to_rest_qpos"]) * info["is_item_grasped"].float()


def fingers_touching_table(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 if a finger touches the table (use with a negative weight)."""
    return lift_info(env)["robot_touching_table"].float()


def cube_not_lifted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 while the cube is not lifted (use with a negative weight, encourages lifting fast)."""
    return (~lift_info(env)["item_lifted"]).float()


class lift_success_metrics(ManagerTermBase):
    """Not a reward: register with ``weight=0.0``. Logs squint's ``success_once`` / ``success_at_end`` per
    episode (``Metrics/...`` in ``extras["log"]``) and clears the per-episode buffers on reset."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor | None = None):
        bufs = _episode_buffers(self._env)
        if env_ids is None:
            env_ids = slice(None)
        log = self._env.extras.setdefault("log", {})
        log["Metrics/success_once"] = bufs["success_once"][env_ids].float().mean().item()
        log["Metrics/success_at_end"] = bufs["success"][env_ids].float().mean().item()
        bufs["cube_rest_z"][env_ids] = float("inf")
        bufs["success"][env_ids] = False
        bufs["success_once"][env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.zeros(env.num_envs, device=env.device)
