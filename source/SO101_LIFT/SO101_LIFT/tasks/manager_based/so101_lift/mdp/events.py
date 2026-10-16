from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_gaussian(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the joints to the default (rest) pose plus Gaussian noise, as in squint's ``_initialize_episode``."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_pos[:, asset_cfg.joint_ids] += torch.randn_like(joint_pos[:, asset_cfg.joint_ids]) * std
    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # like ManiSkill's controller reset: the joint targets start at the new joint positions (otherwise the
    # joint_pos_target observation would show the previous episode's last target in the first step)
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
