"""Created: 2023/6/13"""
from environment import RescueEnv
import gymnasium as gym
from marlgrid.envs import register_marl_env, ClutteredMultiGrid
import time


if __name__ == '__main__':
    register_marl_env(
        "MyEnv",
        ClutteredMultiGrid,
        n_agents=3,
        grid_size=19,
        view_size=5,
        env_kwargs={'clutter_density':0.15, 'randomize_goal':True}
    )
    env = gym.make(
        "MyEnv",
    )
    env = RescueEnv(env)
    env.start()
    env.step([2, 2, 1])
    time.sleep(1)
    env.env.close()
