import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import Conv2d
import gymnasium as gym

# 引入你的环境依赖
from environment import RescueEnv
from marlgrid.envs import register_marl_env, ClutteredMultiGrid

# ----------------------------------------------------------------
# 必须保持与训练时完全一致的 Actor 定义
# ----------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv_1 = Conv2d(1, 1, 5)
        self.conv_2 = Conv2d(1, 1, 3)
        self.conv_3 = Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(121, 121)

        self.conv_local = Conv2d(1, 1, 3, padding=1)
        self.fc2 = nn.Linear(25, 25)

        self.fc = nn.Linear(146, 3)

    def forward(self, s0, si, action=None, deterministic=False):
        # s0: [bs, c, 19, 19]
        bs = s0.shape[0]
        s0_1 = F.relu(self.conv_1(s0))
        s0_2 = F.relu(self.conv_2(s0_1))
        s0_3 = F.relu(self.conv_3(s0_2))
        s0_out = F.relu(self.fc1(s0_3.view(bs, 121)))

        si_1 = F.relu(self.conv_local(si))
        si_out = F.relu(self.fc2(si_1.view(bs, 25)))

        s = torch.cat((s0_out, si_out), 1)
        logits = self.fc(s)
        
        if deterministic:
            # 评估模式：选择概率最大的动作
            a = torch.argmax(logits, dim=1)
            return a, None
        else:
            # 随机模式
            dist = Categorical(logits=logits)
            if action is None:
                a = dist.sample()
            else:
                a = action
            return a, None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
        help="path to the saved actor.pth file")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--num-episodes", type=int, default=100,
        help="number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
        help="whether to render the environment")
    parser.add_argument("--max-steps", type=int, default=400,
        help="max steps per episode")
    parser.add_argument("--deterministic", action="store_true", default=True,
        help="use deterministic policy (argmax) instead of sampling")
    return parser.parse_args()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 环境设置 (必须与训练时参数一致)
    register_marl_env(
        "MyEvalEnv",
        ClutteredMultiGrid,
        n_agents=3,
        grid_size=19,
        view_size=5,
        env_kwargs={
            'clutter_density': 0.15, 
            'randomize_goal': True,
            'max_steps': args.max_steps
        }
    )
    
    # 注意：如果开启渲染，render_mode 需要设为 human
    # 但你的 RescueEnv 封装逻辑比较特殊，这里先使用默认 make，然后在 loop 中调用 env.render()
    raw_env = gym.make("MyEvalEnv", disable_env_checker=True)
    env = RescueEnv(raw_env)

    # 2. 加载模型
    actor = Actor().to(device)
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        actor.load_state_dict(state_dict)
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    actor.eval() # 切换到评估模式

    total_rewards = []
    success_count = 0
    step_counts = []

    for episode in range(args.num_episodes):
        obs = env.reset(seed=args.seed + episode)
        
        # --- 数据预处理 (与训练代码完全一致) ---
        map_np = env.get_s0()
        next_map = torch.as_tensor(map_np, dtype=torch.float32).to(device)
        if len(next_map.shape) == 2:
            next_map = next_map.unsqueeze(0) # [1, 19, 19]

        obs_list = np.array(env.get_s_agent())
        next_local_obs = torch.as_tensor(obs_list, dtype=torch.float32).to(device)
        if len(next_local_obs.shape) == 3: 
            next_local_obs = next_local_obs.unsqueeze(1) # [3, 1, 5, 5]

        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            steps += 1
            if args.render:
                env.env.render()
                time.sleep(0.05) # 控制播放速度

            with torch.no_grad():
                actions = torch.zeros(env.agents).to(device)
                
                for i in range(env.agents):
                    # 使用 deterministic=args.deterministic
                    action, _ = actor(
                        next_map.unsqueeze(0), 
                        next_local_obs[i].unsqueeze(0),
                        deterministic=args.deterministic
                    )
                    actions[i] = action

            # Step
            # 注意：这里的 unpacking 取决于你是否已经在 environment.py 中做了兼容
            # 如果没做兼容，这里可能需要调整
            step_result = env.step(actions)
            if len(step_result) == 5:
                _, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                _, reward, done, _ = step_result
                terminated = done # 假设旧接口

            episode_reward += np.sum(reward) # 统计 Team Reward

            # Update Obs
            map_np = env.get_s0()
            next_map = torch.as_tensor(map_np, dtype=torch.float32).to(device)
            if len(next_map.shape) == 2:
                next_map = next_map.unsqueeze(0)
            
            obs_list = np.array(env.get_s_agent())
            next_local_obs = torch.as_tensor(obs_list, dtype=torch.float32).to(device)
            if len(next_local_obs.shape) == 3:
                next_local_obs = next_local_obs.unsqueeze(1)

        # 记录统计数据
        total_rewards.append(episode_reward)
        step_counts.append(steps)
        
        # 判断是否成功 (通过 map 中是否有 'G' 来判断，或者通过 reward)
        # 简单判断：如果不是因为超时结束的，通常意味着找到了目标
        # 更严谨的判断：
        is_success = ('G' not in str(env.env.unwrapped.grid))
        if is_success:
            success_count += 1

        print(f"Episode {episode+1}/{args.num_episodes} | Steps: {steps} | Reward: {episode_reward:.2f} | Success: {is_success}")

    env.close()

    print("\n" + "="*30)
    print(f"Evaluation Results ({args.num_episodes} episodes)")
    print(f"Success Rate: {success_count/args.num_episodes*100:.2f}%")
    print(f"Avg Reward:   {np.mean(total_rewards):.2f}")
    print(f"Avg Steps:    {np.mean(step_counts):.2f}")
    print("="*30)

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)