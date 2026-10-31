# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug the grasp detection of the SO-101 lift task.

Runs a few episodes and prints, for one env and every policy step:
- distance TCP -> cube, gripper joint angle and target,
- cube and TCP positions in the fixed-jaw frame (where things are relative to the gripper), fingertip gap,
- contact force from the cube on each finger, the angle between the two forces, other (table) forces,
- the grasp / table / lifted flags used by the reward.
After each episode it prints statistics over all envs (how often the fingers touch / squeeze the cube).
The two fingertip frames (the TCP is their midpoint) are shown as markers in the viewer.

Examples:
    # trained policy
    python scripts/rl_games/debug_grasp.py --task Template-So101-Lift-v0 --checkpoint <path to .pth>
    # arm still, gripper opening and closing (check the fingertip markers and the gap)
    python scripts/rl_games/debug_grasp.py --task Template-So101-Lift-v0 --actions gripper_sweep
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug the grasp detection of the SO-101 lift task.")
parser.add_argument("--task", type=str, default="Template-So101-Lift-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (needed for --actions policy).")
parser.add_argument(
    "--actions",
    type=str,
    default="policy",
    choices=["policy", "zero", "gripper_sweep"],
    help="policy: trained checkpoint, zero: no motion, gripper_sweep: arm still, gripper opens and closes.",
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--debug_env", type=int, default=0, help="Index of the env whose per-step values are printed.")
parser.add_argument("--num_episodes", type=int, default=3, help="Number of episodes to run.")
parser.add_argument("--marker_scale", type=float, default=0.05, help="Size of the TCP frame marker.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import SO101_LIFT.tasks  # noqa: F401
from SO101_LIFT.assets.so101 import SO101_END_EFFECTOR, SO101_FINGER_BODIES, SO101_FIXED_JAW, SO101_GRIPPER_JOINTS
from SO101_LIFT.tasks.manager_based.so101_lift.mdp import observations as lift_obs
from SO101_LIFT.tasks.manager_based.so101_lift.mdp import rewards as lift_rewards

TOUCH_FORCE = 0.05  # N, "finger touches the cube" for the statistics


def fmt(v: torch.Tensor, scale: float = 100.0) -> str:
    """Format a 3D vector in cm."""
    return "({:6.1f},{:6.1f},{:6.1f})".format(*(v * scale).tolist())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict):
    if args_cli.actions == "policy" and args_cli.checkpoint is None:
        raise ValueError("--actions policy needs --checkpoint (or use --actions zero / gripper_sweep)")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    # show the fingertip frames (ee_frame) in the viewer
    env_cfg.scene.ee_frame.debug_vis = True
    env_cfg.scene.ee_frame.visualizer_cfg.markers["frame"].scale = (args_cli.marker_scale,) * 3

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    agent = None
    if args_cli.actions == "policy":
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        runner = Runner()
        runner.load(agent_cfg)
        agent = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

    isaac_env = env.unwrapped
    robot = isaac_env.scene["robot"]
    cube = isaac_env.scene["cube"]
    ee_frame = isaac_env.scene["ee_frame"]
    sensor = isaac_env.scene["robot_contacts"]
    num_envs = isaac_env.num_envs
    n_act = isaac_env.action_manager.total_action_dim
    max_steps = isaac_env.max_episode_length
    e = args_cli.debug_env

    fixed_id = robot.find_bodies(SO101_FIXED_JAW)[0][0]
    jaw_id = robot.find_joints(SO101_GRIPPER_JOINTS)[0][0]
    finger_ids = [sensor.body_names.index(n) for n in SO101_FINGER_BODIES]

    print("\n================ SO-101 grasp debug ================")
    print(f"actions: {args_cli.actions}, envs: {num_envs}, printed env: {e}, episode: {max_steps} steps")
    print(f"robot bodies: {robot.body_names}")
    print(f"contact sensor bodies: {sensor.body_names}")
    print(f"fingers: fixed jaw '{SO101_FIXED_JAW}', moving jaw '{SO101_END_EFFECTOR}' (sensor ids {finger_ids})")
    lim = robot.data.soft_joint_pos_limits[0, jaw_id]
    print(
        f"gripper joint '{SO101_GRIPPER_JOINTS[0]}': limits [{lim[0]:.3f}, {lim[1]:.3f}] rad, "
        f"default {robot.data.default_joint_pos[0, jaw_id]:.3f} rad, "
        f"stiffness {robot.data.joint_stiffness[0, jaw_id]:.2f}"
    )
    print("positions below are in cm in the FIXED-JAW frame; forces in N; flags G=grasped T=table L=lifted")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    if agent is not None:
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

    for episode in range(args_cli.num_episodes):
        touch_both = torch.zeros(num_envs, device=isaac_env.device)
        touch_one = torch.zeros_like(touch_both)
        squeeze_both = torch.zeros_like(touch_both)
        grasped = torch.zeros_like(touch_both)
        max_min_force = torch.zeros_like(touch_both)

        print(f"\n--- episode {episode} ---")
        print(
            "step | tcp-cube | jaw q / target |      cube@jaw      |      tcp@jaw       | gap "
            "| F_cube fix/mov |  cos  | F_other fix/mov | G T L"
        )
        for t in range(max_steps):
            if not simulation_app.is_running():
                return
            with torch.inference_mode():
                if agent is not None:
                    actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
                else:
                    actions = torch.zeros(num_envs, n_act, device=rl_device)
                    if args_cli.actions == "gripper_sweep":
                        actions[:, -1] = math.sin(2 * math.pi * t / 20)
                obs, _, _, _ = env.step(actions)
                if isinstance(obs, dict):
                    obs = obs["obs"]

                # flags of this step, as computed for the reward (before a possible reset)
                info = lift_rewards.lift_info(isaac_env)

                fm = sensor.data.force_matrix_w
                if fm is None:
                    print("force_matrix_w is None: the contact sensor's cube filter is not working!")
                    return
                f_cube = fm[:, finger_ids, 0, :]
                f_cube_norm = torch.linalg.norm(f_cube, dim=-1)
                f_other = torch.linalg.norm(sensor.data.net_forces_w[:, finger_ids, :] - f_cube, dim=-1)
                cos = torch.nn.functional.cosine_similarity(f_cube[:, 0], f_cube[:, 1], dim=-1)

                jaw_pos_w = robot.data.body_pos_w[:, fixed_id]
                jaw_quat_w = robot.data.body_quat_w[:, fixed_id]
                cube_j, _ = subtract_frame_transforms(jaw_pos_w, jaw_quat_w, cube.data.root_pos_w)
                tcp_j, _ = subtract_frame_transforms(jaw_pos_w, jaw_quat_w, lift_obs.tcp_pos_w(isaac_env))
                tips = ee_frame.data.target_pos_w
                gap = torch.linalg.norm(tips[:, 0] - tips[:, 1], dim=-1)

                touching = f_cube_norm > TOUCH_FORCE
                touch_both += touching.all(dim=-1).float()
                touch_one += (touching.any(dim=-1) & ~touching.all(dim=-1)).float()
                squeeze_both += (f_cube_norm >= lift_rewards.GRASP_MIN_FORCE).all(dim=-1).float()
                grasped += info["is_item_grasped"].float()
                max_min_force = torch.maximum(max_min_force, f_cube_norm.min(dim=-1).values)

                print(
                    f"{t:4d} | {info['tcp_to_cube_dist'][e] * 100:6.1f}cm "
                    f"| {robot.data.joint_pos[e, jaw_id]:6.3f} / {robot.data.joint_pos_target[e, jaw_id]:6.3f} "
                    f"| {fmt(cube_j[e])} | {fmt(tcp_j[e])} | {gap[e] * 100:4.1f}cm "
                    f"| {f_cube_norm[e, 0]:6.2f} {f_cube_norm[e, 1]:6.2f} | {cos[e]:5.2f} "
                    f"| {f_other[e, 0]:6.2f} {f_other[e, 1]:6.2f}  "
                    f"| {int(info['is_item_grasped'][e])} {int(info['robot_touching_table'][e])} "
                    f"{int(info['item_lifted'][e])}"
                )

        n = float(max_steps)
        print(f"--- episode {episode} summary over all {num_envs} envs (share of steps) ---")
        print(f"only one finger touches the cube : {(touch_one / n).mean():.3f}")
        print(f"both fingers touch the cube      : {(touch_both / n).mean():.3f}")
        print(f"both fingers >= {lift_rewards.GRASP_MIN_FORCE} N            : {(squeeze_both / n).mean():.3f}")
        print(f"grasp detected (reward)          : {(grasped / n).mean():.3f}")
        print(f"largest weaker-finger force      : {max_min_force.max():.3f} N (per env: {max_min_force.tolist()})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
