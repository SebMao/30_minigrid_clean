import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Conv2d

# 确保这些环境文件在你的项目目录下存在
from environment import RescueEnv
from gymnasium import Env
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
from marlgrid.envs import register_marl_env, ClutteredMultiGrid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="MAAttack",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    # Modified: max steps per episode instead of fixed update rollout length
    parser.add_argument("--max-episode-steps", type=int, default=200,
        help="the maximum number of steps per episode")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    return args

class RolloutBuffer:
    def __init__(self):
        self.maps = []
        self.local_obses = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []

    def clear(self):
        self.maps = []
        self.local_obses = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = []
        self.returns = []

    def add(self, state_map, state_local, action, logprob, reward, done, value):
        # state_map: [1, C, H, W] -> remove batch dim for storage to save memory? 
        # Actually keeping it [1, C, H, W] is fine if we cat later.
        self.maps.append(state_map)           
        self.local_obses.append(state_local)  # [Agents, C, H, W]
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_value, next_done, gamma, gae_lambda):
        # device check based on next_value
        device = next_value.device
        
        # self.rewards 是 [[r1, r2, r3], ...] -> tensor 变成 [T, 3]
        rewards = torch.tensor(self.rewards).to(device)
        
        # Check if values list is empty or not
        if len(self.values) == 0:
            return 
            
        # self.values 是 [tensor([v]), ...] -> cat 变成 [T+1]
        values = torch.cat(self.values + [next_value.unsqueeze(0)]).to(device)
        dones = torch.tensor(self.dones).to(device).float()
        
        if not isinstance(next_done, torch.Tensor):
            # 显式指定 dtype=torch.float32，防止生成 BoolTensor
            next_done = torch.tensor(next_done, dtype=torch.float32).to(device)
        else:
            # 如果已经是 Tensor，确保它是 float 类型
            next_done = next_done.float()

        episode_len = len(self.rewards)
        
        # advantages 形状跟随 rewards: [T, 3]
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        
        for t in reversed(range(episode_len)):
            if t == episode_len - 1:
                nextnonterminal = 1.0 - next_done
                nextvals = values[t + 1]
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvals = values[t + 1]
            
            # loop 中自动广播: rewards[t]([3]) + scalar - scalar = [3]
            delta = rewards[t] + gamma * nextvals * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        # === 修复点在这里 ===
        # advantages [T, 3] + values[:-1] [T] -> 会报错
        # values[:-1].unsqueeze(-1) 变成 [T, 1] -> 广播成功
        returns = advantages + values[:-1].unsqueeze(-1)
        
        self.advantages = advantages
        self.returns = returns

    def get_batch(self):
        if len(self.maps) == 0:
            return None
        
        # --- 修复点：使用 stack 而不是 cat ---
        # self.maps 是 [Tensor(1, 19, 19), Tensor(1, 19, 19)...]
        # torch.cat -> [T, 19, 19] (错误，丢失 Channel)
        # torch.stack -> [T, 1, 19, 19] (正确)
        b_maps = torch.stack(self.maps) 
        
        # 其他保持不变
        b_local_obses = torch.stack(self.local_obses) 
        b_actions = torch.stack(self.actions) 
        b_logprobs = torch.stack(self.logprobs) 
        b_advantages = self.advantages
        b_returns = self.returns
        b_values = torch.cat(self.values) 
        
        return b_maps, b_local_obses, b_actions, b_logprobs, b_advantages, b_returns, b_values

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

    def forward(self, s0, si, action=None):
        # s0: [bs, c, 19, 19]
        # si: [bs, c, 5, 5] (single agent view)
        bs = s0.shape[0]
        s0_1 = F.relu(self.conv_1(s0))
        s0_2 = F.relu(self.conv_2(s0_1))
        s0_3 = F.relu(self.conv_3(s0_2))
        s0_out = F.relu(self.fc1(s0_3.view(bs, 121)))

        si_1 = F.relu(self.conv_local(si))
        si_out = F.relu(self.fc2(si_1.view(bs, 25)))

        s = torch.cat((s0_out, si_out), 1)
        logits = self.fc(s)
        dist = Categorical(logits=logits)
        if action is None:
            a = dist.sample()
        else:
            a = action
        logprob = dist.log_prob(a.squeeze())
        return a, logprob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv_1 = Conv2d(1, 1, 5)
        self.conv_2 = Conv2d(1, 1, 3)
        self.conv_3 = Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(121, 121)

        self.conv_local = Conv2d(1, 3, 3, padding=1)
        self.fc2 = nn.Linear(75, 25)

        self.fc = nn.Linear(196, 1)

    def forward(self, s0, sis):
        # s0: [bs, c, 19, 19]
        # sis: [bs, agents, c, 5, 5]
        s0_1 = F.relu(self.conv_1(s0))
        s0_2 = F.relu(self.conv_2(s0_1))
        s0_3 = F.relu(self.conv_3(s0_2))
        s0_out = F.relu(self.fc1(s0_3.view(-1, 121)))

        ss = []
        for id_ in range(sis.shape[1]):
            # sis[:, id_] -> [bs, c, 5, 5]
            si = sis[:, id_] 
            si_1 = F.relu(self.conv_local(si))
            si_out = F.relu(self.fc2(si_1.view(-1, 75)))
            ss.append(si_out)
        si_out = torch.cat(ss, 1)

        s = torch.cat([s0_out, si_out], 1)
        v = self.fc(s)

        return v.squeeze()

def main(args):
    torch.autograd.set_detect_anomaly(True)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env Setup
    register_marl_env(
        "MyEnv",
        ClutteredMultiGrid,
        n_agents=3,
        grid_size=19,
        view_size=5,
        env_kwargs={
            'clutter_density': 0.15, 
            'randomize_goal': True,
            'max_steps': args.max_episode_steps  # <--- 添加这一行
        }
    )
    raw_env = gym.make("MyEnv")
    env = RescueEnv(raw_env)

    actor = Actor().to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
    critic = Critic().to(device)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)

    buffer = RolloutBuffer()
    
    global_step = 0
    
    pbar = tqdm(total=args.total_timesteps)
    
    while global_step < args.total_timesteps:
        # Reset Env
        env.reset()
        
        # --- Preprocessing Step 1 ---

        
        map_np = env.get_s0()
        next_map = torch.as_tensor(map_np, dtype=torch.float32).to(device)        # Ensure map has correct batch dim for storage/forward if needed, 
        # or just treat [C, H, W] as one sample. 
        # Code used `maps[step].unsqueeze(0).unsqueeze(0)` previously implying `maps` was `[19, 19]`.
        # Let's standardize: next_map -> [1, 19, 19] (Channel, H, W)
        if len(next_map.shape) == 2:
            next_map = next_map.unsqueeze(0)

        # Agent Obs: list of [5, 5] -> [3, 5, 5]
        # We need [Agents, Channel, H, W] -> [3, 1, 5, 5]
        obs_list = np.array(env.get_s_agent())
        next_local_obs = torch.as_tensor(obs_list, dtype=torch.float32).to(device)
        if len(next_local_obs.shape) == 3: # [3, 5, 5]
            next_local_obs = next_local_obs.unsqueeze(1) # [3, 1, 5, 5]

        episode_step = 0
        done = False
        
        # Episode Loop
        while not done and episode_step < args.max_episode_steps:
            episode_step += 1
            global_step += 1 
            
            # Anneal LR
            if args.anneal_lr:
                frac = 1.0 - (global_step - 1.0) / args.total_timesteps
                lrnow = frac * args.learning_rate
                actor_optimizer.param_groups[0]["lr"] = lrnow
                critic_optimizer.param_groups[0]["lr"] = lrnow

            with torch.no_grad():
                actions = torch.zeros(env.agents).to(device)
                logprobs = torch.zeros(env.agents).to(device)
                
                # --- Actor Inference ---
                for i in range(env.agents):
                    # next_local_obs[i] is [1, 5, 5]
                    # Actor expects [BS, C, H, W]
                    # Pass [1, 1, 19, 19] and [1, 1, 5, 5]
                    action, logprob = actor(
                        next_map.unsqueeze(0),        # [1, 1, 19, 19]
                        next_local_obs[i].unsqueeze(0) # [1, 1, 5, 5]
                    )
                    actions[i] = action
                    logprobs[i] = logprob
                
                # --- Critic Inference ---
                # Critic expects [BS, Agents, C, H, W]
                # next_local_obs is [3, 1, 5, 5] -> Unsqueeze(0) -> [1, 3, 1, 5, 5]
                value = critic(
                    next_map.unsqueeze(0), 
                    next_local_obs.unsqueeze(0)
                )

            # Step Env
            obs, reward, terminated, truncated, info = env.step(actions)

            done = terminated or truncated
            
            # Add to buffer
            buffer.add(
                state_map=next_map, 
                state_local=next_local_obs, 
                action=actions, 
                logprob=logprobs, 
                reward=reward, 
                done=done, 
                value=value.view(-1)
            )
            
            # Update Next Obs
            map_np = env.get_s0()
            next_map = torch.as_tensor(map_np, dtype=torch.float32).to(device)
            if len(next_map.shape) == 2:
                next_map = next_map.unsqueeze(0)
            
            obs_list = np.array(env.get_s_agent())
            next_local_obs = torch.as_tensor(obs_list, dtype=torch.float32).to(device)
            if len(next_local_obs.shape) == 3:
                next_local_obs = next_local_obs.unsqueeze(1) # [3, 1, 5, 5]
            
            pbar.update(1)

        # --- End of Episode: Compute GAE and Update ---
        
        # Bootstrap value
        with torch.no_grad():
            next_value = critic(
                next_map.unsqueeze(0), 
                next_local_obs.unsqueeze(0)
            )
        
        buffer.compute_gae(next_value.squeeze(), done, args.gamma, args.gae_lambda)
        
        # Get Batch
        batch_data = buffer.get_batch()
        if batch_data is None: 
            continue
            
        b_maps, b_local_obses, b_actions, b_logprobs, b_advantages, b_returns, b_values = batch_data
        
        # Update Loop
        # Batch size is current episode length
        batch_size = b_maps.shape[0]
        if batch_size < args.num_minibatches:
            # Skip update if episode is too short for batching, or adjust minibatch
            # Here we simply use the whole episode as one batch if it's small
            minibatch_size = batch_size
        else:
            minibatch_size = int(batch_size // args.num_minibatches)
            
        b_inds = np.arange(batch_size)
        
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                newlogprobs = torch.zeros(len(mb_inds), env.agents).to(device)
                
                for i in range(env.agents):
                    _, newlogprob = actor(
                        b_maps[mb_inds], 
                        b_local_obses[mb_inds, i], 
                        b_actions[mb_inds, i]
                    )
                    newlogprobs[:, i] = newlogprob

                newvalue = critic(
                    b_maps[mb_inds], 
                    b_local_obses[mb_inds]
                )

                logratio = newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # [已删除] mb_advantages = mb_advantages.unsqueeze(1).repeat(1, env.agents)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                # 注意：这里 b_returns 是 [Batch, 3]，newvalue 是 [Batch]
                # 这会导致 newvalue 试图同时逼近 3 个不同的 return。
                # 如果你想让 Critic 预测 Team Reward，这里可能需要取 b_returns 的平均或和。
                # 但为了先跑通，直接计算也是可以的（会触发 Broadcasting），Critic 会学到平均值。
                
                # 为了防止维度警告，建议把 b_returns 展平或者取平均，这里暂时保持广播形式：
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue.unsqueeze(1) - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds].unsqueeze(1) + torch.clamp(
                        newvalue.unsqueeze(1) - b_values[mb_inds].unsqueeze(1),
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # newvalue: [BS], b_returns: [BS, 3] -> newvalue.unsqueeze(1): [BS, 1]
                    v_loss = 0.5 * ((newvalue.unsqueeze(1) - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss + v_loss * args.vf_coef

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

        # Clear buffer
        buffer.clear()
        
        # Logging
        if len(b_returns) > 0:
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("charts/episode_len", episode_step, global_step)
            writer.add_scalar("charts/episode_reward", sum(buffer.rewards) if len(buffer.rewards)>0 else 0, global_step)
            if global_step % 1000 == 0:
                print(f"Step: {global_step}, EpLen: {episode_step}, Loss: {loss.item():.3f}")

    env.close()
    writer.close()
    pbar.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)