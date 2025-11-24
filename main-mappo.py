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

from environment import RescueEnv
from gymnasium import Env
from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
import time
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
    parser.add_argument("--num-steps", type=int, default=200,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.95,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=10,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,  # original: 0.01
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--delta_", type=float, default=2)
    parser.add_argument("--delta_length", type=float, default=8)
    parser.add_argument("--stage2", type=bool, default=True)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

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
            si = sis[:, id_]
            si_1 = F.relu(self.conv_local(si))
            si_out = F.relu(self.fc2(si_1.view(-1, 75)))
            ss.append(si_out)
        si_out = torch.cat(ss, 1)

        s = torch.cat([s0_out, si_out], 1)
        v = self.fc(s)

        return v.squeeze()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def main(args):
    flag = 0
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

    device = "cpu"

    # env setup
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

    actor = Actor()
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate, eps=1e-5)
    critic = Critic()
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup

    maps = torch.zeros(args.num_steps, env.map_size, env.map_size)
    local_obses = torch.zeros(args.num_steps, env.agents, env.view_size, env.view_size)

    actions = torch.zeros(args.num_steps, env.agents)
    logprobs = torch.zeros(args.num_steps, env.agents).to(device)
    rewards = torch.zeros(args.num_steps, env.agents).to(device)
    dones = torch.zeros(args.num_steps).to(device)
    values = torch.zeros(args.num_steps).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    env.reset()
    next_map = torch.Tensor(env.get_s0())
    next_local_obs = torch.Tensor(env.get_s_agent())

    next_done = torch.zeros(1).to(device)
    num_updates = args.total_timesteps // args.batch_size
    history_J = np.array([])

    for update in tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            actor_optimizer.param_groups[0]["lr"] = lrnow
            critic_optimizer.param_groups[0]["lr"] = lrnow
        # if update >= 2000:
        #     print('theta changed')
        #     args.theta_coef = 5

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            maps[step] = next_map
            local_obses[step] = next_local_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # with torch.no_grad():
            with torch.no_grad():
                for i in range(env.agents):
                    action, logprob = actor(maps[step].unsqueeze(0).unsqueeze(0),
                                            local_obses[step][i].unsqueeze(0).unsqueeze(0))
                    actions[step][i] = action
                    logprobs[step][i] = logprob
                value = critic(maps[step].unsqueeze(0).unsqueeze(0), local_obses[step].unsqueeze(1).unsqueeze(0))
                values[step] = value

            # TRY NOT TO MODIFY: execute the game and log data.
            _, reward, done, _ = env.step(actions[step])
            next_map = env.get_s0()
            next_local_obs = env.get_s_agent()

            # if done[0] == 1:
            #     with torch.no_grad():
            #         value_ = agent.get_value(info)
            #         reward = reward + args.gamma * value_
            rewards[step] = torch.tensor(reward)
            next_done = torch.Tensor([done])

        # bootstrap value if not done
        with torch.no_grad():
            next_value = critic(next_map.unsqueeze(0).unsqueeze(0), next_local_obs.unsqueeze(1).unsqueeze(0))
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values.unsqueeze(1)
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    # returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return + args.gamma * (1-nextnonterminal) * next_value
                    # next_value = values[t]
                advantages = returns - values

        print("Training ....")
        b_maps = maps
        b_local_obses = local_obses
        b_logprobs = logprobs
        b_actions = actions
        b_advantages = advantages
        b_returns = returns.sum(1)
        b_values = values

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                newlogprob = torch.zeros(args.minibatch_size, env.agents)  # BS * #agents
                mb_inds = b_inds[start:end]
                for i in range(env.agents):
                    _, newlogprob[:, i] = actor(b_maps[mb_inds].unsqueeze(1), b_local_obses[mb_inds, i].unsqueeze(1), b_actions[mb_inds, i].unsqueeze(1))
                newvalue = critic(b_maps[mb_inds].unsqueeze(1), b_local_obses[mb_inds].unsqueeze(2))

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # mb_advantages = b_advantages[mb_inds]
                # if args.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                #
                # # Policy loss
                # pg_loss1 = -mb_advantages * ratio
                # pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)

                mb_advantages = returns[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)

                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()


                loss = pg_loss + v_loss * args.vf_coef


                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

            # if args.target_kl is not None:
            #     if approx_kl > args.target_kl:
            #         break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/returns", torch.sum(rewards), global_step)
        print(torch.sum(rewards), global_step, v_loss.item())
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if global_step % 200000 == 0:
            print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    env.close()
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
