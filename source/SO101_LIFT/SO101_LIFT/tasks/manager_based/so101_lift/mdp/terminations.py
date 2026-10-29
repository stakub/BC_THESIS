from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def invalid_state(
    env: ManagerBasedRLEnv,
    max_cube_offset: float = 1.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Reset envs whose physics blew up, before their NaN / huge values reach the RL algorithm.

    True if a robot joint state / joint target or the cube state is not finite, or if the cube is further than
    ``max_cube_offset`` metres from its spawn position (it was flung away or fell off the table).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    bad = ~torch.isfinite(robot.data.joint_pos).all(dim=-1)
    bad |= ~torch.isfinite(robot.data.joint_vel).all(dim=-1)
    bad |= ~torch.isfinite(robot.data.joint_pos_target).all(dim=-1)  # e.g. a NaN IK solution
    bad |= ~torch.isfinite(cube.data.root_state_w).all(dim=-1)
    spawn_pos = cube.data.default_root_state[:, :3] + env.scene.env_origins
    bad |= torch.linalg.norm(cube.data.root_pos_w - spawn_pos, dim=-1) > max_cube_offset
    return bad
