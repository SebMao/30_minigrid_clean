"""Created: 2023/6/13"""
from __future__ import annotations

import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from typing import List
import torch
import marlgrid.envs
import numpy as np
from marlgrid.envs import MultiGridEnv
pygame.init()


class RescueEnv:
    def __init__(
        self,
        env: MultiGridEnv,
        seed=None,
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.agents = 3
        self.map_size = 19
        self.view_size = 5
        self.action_space = 1


    def grid_str2array(self, grid):
        # 墙壁WW 1; 空地 2; 受害者 GG  3; 当前位置NESW >> 4567; 未知 0
        grid = str(grid).split('*\n*')
        grid = grid[1:-1]
        map0 = {
            "WW": 1,
            "  ": 2,
            "GG": 3,
            "^^": 4,
            ">>": 5,
            "VV": 6,
            "<<": 7,
        }
        m, n = len(grid), len(grid[0])//2
        ret = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(0, n):
                ret[i][j] = map0[grid[i][2*j:2*j+2]]
        ret = np.array(ret)
        ret = (ret - 3) / 5  # 归一化
        return ret

    def get_s0(self):
        # 获得地图信息
        grid = self.env.grid
        ret = self.grid_str2array(grid)
        viewed_area = self.env.viewed_area


        return torch.Tensor(ret * viewed_area * 0)  # TODO: 关闭全局地图的实验


    def get_s_agent(self):
        agent_views = []
        for a in self.env.agents:
            obs, _ = self.env.gen_obs_grid(a)
            agent_views.append(self.grid_str2array(obs))
        return torch.Tensor(agent_views)

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)


    def step(self, action: List[Actions]):
        obs, rewards, terminated, _ = self.env.step(action)
        # print(f"step={self.env.step_count}, reward={rewards}")

        if terminated:
            print("terminated!")
            self.env.render()
            # self.reset(self.seed)
        else:
            self.env.render()
        return obs, rewards, terminated, _

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        # default="MiniGrid-Empty-8x8-v0",
        default="MarlGrid-3AgentCluttered15x15-v0",
        # default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        # grid_size=20
        # tile_size=args.tile_size,
        # render_mode="human",
        # agent_pov=args.agent_view,
        # agent_view_size=args.agent_view_size,
        # screen_size=args.screen_size,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    manual_control = ManualControl(env, seed=args.seed)
    manual_control.start()
