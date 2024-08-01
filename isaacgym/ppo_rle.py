# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
# Same as the other file except we train on the true reward instead of the shaped reward
# Only change is: use true_rewards instead of rewards when computing advantages and returns
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import isaacgym  # noqa
import isaacgymenvs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from collections import deque
from gym.wrappers.normalize import RunningMeanStd
from copy import deepcopy


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="random-latent-exploration",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--gpu-id", type=int, default=0,
        help="ID of GPU to use")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Ant",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=30000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=0.001,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4096,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=16,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=2,
        help="coefficient of the value function")
    parser.add_argument("--int-vf-coef", type=float, default=0.5,
        help="coefficient of the intrinsic value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    parser.add_argument("--reward-scaler", type=float, default=1,
        help="the scale factor applied to the reward during training")
    parser.add_argument("--record-video-step-frequency", type=int, default=1464,
        help="the frequency at which to record the videos")

    # RLE arguments
    parser.add_argument("--switch-steps", type=int, default=16,
        help="number of timesteps to switch the RLE network")
    parser.add_argument("--norm-rle-features", type=lambda x: bool(strtobool(x)), default=False,
                        nargs="?", const=True, help="if toggled, rle features will be normalized")
    parser.add_argument("--int-coef", type=float, default=0.1,
        help="coefficient of extrinsic reward")
    parser.add_argument("--ext-coef", type=float, default=1.0,
        help="coefficient of intrinsic reward")
    parser.add_argument("--int-gamma", type=float, default=0.99,
        help="Intrinsic reward discount rate")
    parser.add_argument("--feature-size", type=int, default=32,
        help="Size of the feature vector output by the rle network")
    parser.add_argument("--tau", type=float, default=0.005,
        help="The parameter for soft updating the rle network")
    parser.add_argument("--save-rle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, save the rle network at the end of training")
    parser.add_argument("--num-iterations-feat-norm-init", type=int, default=1,
        help="number of iterations to initialize the feature normalization parameters")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ground_truth_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_ground_truth_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.ground_truth_episode_returns += infos["true_rewards"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_ground_truth_episode_returns[:] = self.ground_truth_episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - dones.float()
        self.ground_truth_episode_returns *= 1 - dones.float()
        self.episode_lengths *= 1 - dones.float()
        infos["r"] = self.returned_episode_returns
        infos["ground_truth_r"] = self.returned_ground_truth_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, rle_net):
        super().__init__()
        self.goal_embed_size = 16
        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(rle_net.feature_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.goal_embed_size), std=0.1),
            nn.Tanh(),
        )
        self.goal_state_encoder = nn.Sequential(
            layer_init(nn.Linear(256 + self.goal_embed_size, 256)),
            nn.Tanh(),
        )
        self.critic_base = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
        )
        self.critic_int = layer_init(nn.Linear(256, 1), std=1.0)
        self.critic_ext = layer_init(nn.Linear(256, 1), std=1.0)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + self.goal_embed_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        # make the action standard deviation conditioned on goal embed but not the observation
        self.actor_logstd = nn.Sequential(
            layer_init(nn.Linear(self.goal_embed_size, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=1.0),
        )

    def get_action_and_value(self, x, reward, goal, action=None, deterministic=False):
        goal_embed = self.goal_encoder(goal)

        # Action computation
        action_mean = self.actor_mean(torch.cat([x, goal_embed], dim=1))
        action_logstd = self.actor_logstd(goal_embed)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = probs.sample()

        # Critic computation
        hidden = self.critic_base(x)
        hidden = self.goal_state_encoder(torch.cat([hidden, goal_embed], dim=1))
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic_ext(hidden), self.critic_int(hidden)

    def get_value(self, x, reward, goal):
        goal_embed = self.goal_encoder(goal)
        hidden = self.critic_base(x)
        hidden = self.goal_state_encoder(torch.cat([hidden, goal_embed], dim=1))

        return self.critic_ext(hidden), self.critic_int(hidden)


class RLEModel(nn.Module):
    def __init__(self, envs, feature_size, num_envs, device):
        super().__init__()

        self.device = device

        self.feature_size = feature_size

        # RLE network - this is the same as the "base" network in the agent
        self.rle_net = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 256)),
            nn.Tanh(),
        )
        self.last_layer = layer_init(nn.Linear(256, self.feature_size))

        # Input to the last layer is the feature of the current state, shape = (num_envs, feature_size)
        self.current_ep = 0
        self.num_envs = num_envs
        self.goals = self.sample_goals()

        # num steps left for this goal per environment: switch_steps
        # resets whenever it is 0 or a life ends
        self.num_steps_left = args.switch_steps * torch.ones(num_envs).to(device)
        self.switch_goals_mask = torch.zeros(num_envs).to(device)

        # Maintain statistics for the rle network to be used for normalization
        self.rle_rms = RunningMeanStd(shape=(1, self.feature_size))
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

    def sample_goals(self, num_envs=None):
        if num_envs is None:
            num_envs = self.num_envs
        goals = torch.randn((num_envs, self.feature_size), device=self.device).float()
        return goals

    def step(self, next_done):
        # switch_goals_mask = 0 if the goal is not updated, 1 if the goal is updated
        # switch_goals_mask is a tensor of shape (num_envs,)
        # sample new goals for the environments that need to update their goals
        self.switch_goals_mask = torch.zeros(args.num_envs).to(device)
        self.switch_goals_mask[next_done.bool()] = 1.0
        self.num_steps_left -= 1
        self.switch_goals_mask[self.num_steps_left == 0] = 1.0

        # update the goals
        new_goals = self.sample_goals()
        self.goals = self.goals * (1 - self.switch_goals_mask.unsqueeze(1)) + new_goals * self.switch_goals_mask.unsqueeze(1)

        # update the num_steps_left
        self.num_steps_left[self.switch_goals_mask.bool()] = args.switch_steps

        return self.switch_goals_mask

    def compute_rle_feat(self, obs, goals=None):
        if goals is None:
            goals = self.goals
        with torch.no_grad():
            raw_rle_feat = self.last_layer(self.rle_net(obs))
            rle_feat = (raw_rle_feat - self.rle_feat_mean) / (self.rle_feat_std + 1e-5)
            reward = (rle_feat * goals).sum(axis=1)

        return reward, raw_rle_feat, rle_feat

    def update_rms(self, b_rle_feats):
        # Update the rle rms
        self.rle_rms.update(b_rle_feats)
        # Update the mean and std tensors
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

    def compute_reward(self, obs, next_obs, goals=None):
        return self.compute_rle_feat(next_obs, goals=goals)

    def forward(self, obs, next_obs):
        pass


class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews, not_done=None):
        if not_done is None:
            if self.rewems is None:
                self.rewems = rews
            else:
                self.rewems = self.rewems * self.gamma + rews
            return self.rewems
        else:
            if self.rewems is None:
                self.rewems = rews
            else:
                mask = np.where(not_done == 1.0)
                self.rewems[mask] = self.rewems[mask] * self.gamma + rews[mask]
            return deepcopy(self.rewems)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = isaacgymenvs.make(
        seed=args.seed,
        task=args.env_id,
        num_envs=args.num_envs,
        sim_device=f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu",
        rl_device=f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu",
        graphics_device_id=0 if torch.cuda.is_available() and args.cuda else -1,
        headless=True if torch.cuda.is_available() and args.cuda else True,
        multi_gpu=False,
        virtual_screen_capture=args.capture_video,
        force_render=False,
    )
    if args.capture_video:
        envs.is_vector_env = True
        print(f"record_video_step_frequency={args.record_video_step_frequency}")
        envs = gym.wrappers.RecordVideo(
            envs,
            f"videos/{run_name}",
            step_trigger=lambda step: step % args.record_video_step_frequency == 0,
            video_length=100,  # for each video record up to 100 steps
        )
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    rle_output_size = 8
    rle_network = RLEModel(envs, args.feature_size, args.num_envs, device).to(device)
    rle_feature_size = rle_network.feature_size

    agent = Agent(envs, rle_network).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    int_reward_rms = RunningMeanStd()
    int_discounted_reward = RewardForwardFilter(args.int_gamma)
    ext_reward_rms = RunningMeanStd()
    ext_discounted_reward = RewardForwardFilter(args.gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    goals = torch.zeros((args.num_steps, args.num_envs, rle_feature_size), dtype=torch.float).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)

    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rle_rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    prev_rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    true_rewards = torch.zeros_like(rewards, dtype=torch.float).to(device)

    rle_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)

    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    raw_rle_feats = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)
    rle_feats = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)

    # Logging setup
    num_done_envs = 512
    avg_returns = deque(maxlen=num_done_envs)
    avg_ep_lens = deque(maxlen=num_done_envs)
    avg_consecutive_successes = deque(maxlen=num_done_envs)
    avg_true_returns = deque(maxlen=num_done_envs)  # returns from without reward shaping

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size

    next_raw_rle_feat = []
    # Update rms of the rle network
    if args.norm_rle_features:
        print("Start to initialize rle features normalization parameter.....")
        for step in range(args.num_steps * args.num_iterations_feat_norm_init):
            # Sample random actions from a gaussian distribution and step the environment
            acs = torch.randn((args.num_envs,) + envs.single_action_space.shape, dtype=torch.float).to(device)
            s, r, d, _ = envs.step(acs)
            rle_reward, raw_rle_feat, rle_feat = rle_network.compute_rle_feat(s)
            next_raw_rle_feat += raw_rle_feat.detach().cpu().numpy().tolist()

            if len(next_raw_rle_feat) % (args.num_steps * args.num_envs) == 0:
                next_raw_rle_feat = np.stack(next_raw_rle_feat)
                rle_network.update_rms(next_raw_rle_feat)
                next_raw_rle_feat = []
        print(f"End to initialize... finished in {time.time() - start_time}")

    for update in range(1, num_updates + 1):
        prev_rewards[0] = rle_rewards[-1] * args.int_coef  # last step of prev rollout
        it_start_time = time.time()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done  # done is True if the episode ended in the previous step
            rle_dones[step] = rle_network.switch_goals_mask  # rle_done is True if the goal is switched in the previous step
            goals[step] = rle_network.goals

            # Compute the obs before stepping
            rle_obs = next_obs.clone().float()

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_ext, value_int = agent.get_action_and_value(
                    next_obs, prev_rewards[step], goals[step], deterministic=False
                )
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards[step], next_done, info = envs.step(action)
            true_rewards[step] = info["true_rewards"]

            # Compute the curiosity reward
            rle_rewards[step], raw_rle_feats[step], rle_feats[step] = rle_network.compute_reward(rle_obs, next_obs)

            # update prev rewards
            if step < args.num_steps - 1:
                prev_rewards[step + 1] = rle_rewards[step] * args.int_coef

            for idx, d in enumerate(next_done):
                if d:
                    episodic_return = info["r"][idx].item()
                    true_return = info["ground_truth_r"][idx].item()
                    avg_returns.append(info["r"][idx].item())
                    avg_true_returns.append(info["ground_truth_r"][idx].item())
                    avg_ep_lens.append(info["l"][idx].item())
                    if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                        avg_consecutive_successes.append(info["consecutive_successes"].item())
                    if 0 <= step <= 2:
                        print(f"global_step={global_step}, episodic_return={episodic_return}, true_return={true_return}")

            rle_network.step(next_done)

        not_dones = (1.0 - dones).cpu().data.numpy()
        rewards_cpu = rewards.cpu().data.numpy()
        rle_rewards_cpu = rle_rewards.cpu().data.numpy()

        ext_reward_per_env = np.array(
            [ext_discounted_reward.update(rewards_cpu[i], not_dones[i]) for i in range(args.num_steps)]
        )
        ext_reward_rms.update(ext_reward_per_env.flatten())
        rewards /= np.sqrt(ext_reward_rms.var)

        rle_reward_per_env = np.array(
            [int_discounted_reward.update(rle_rewards_cpu[i], not_dones[i]) for i in range(args.num_steps)]
        )
        int_reward_rms.update(rle_reward_per_env.flatten())
        rle_rewards /= np.sqrt(int_reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs, prev_rewards[step], goals[step])
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(rle_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done.float()
                    int_nextnonterminal = 1.0 - next_done.float()
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0 - rle_dones[t + 1]
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = rle_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_goals = goals.reshape((-1, rle_feature_size))
        b_dones = dones.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_int_values = int_values.reshape(-1)
        b_raw_rle_feats = raw_rle_feats.reshape((-1, rle_feature_size))
        b_prev_rewards = prev_rewards.reshape(-1)

        # Update rms of the rle network
        if args.norm_rle_features:
            rle_network.update_rms(b_raw_rle_feats.cpu().numpy())

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_prev_rewards[mb_inds],
                    b_goals[mb_inds],
                    b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss * args.vf_coef + int_v_loss * args.int_vf_coef
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # update the backbone of the rle network
                for param, target_param in zip(agent.critic_base.parameters(), rle_network.rle_net.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        it_end_time = time.time()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        data = {}
        print("SPS:", int(global_step / (time.time() - start_time)))

        data["charts/iterations"] = update
        data["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
        data["losses/ext_value_loss"] = ext_v_loss.item()
        data["losses/int_value_loss"] = int_v_loss.item()
        data["losses/policy_loss"] = pg_loss.item()
        data["losses/entropy"] = entropy_loss.item()
        data["losses/old_approx_kl"] = old_approx_kl.item()
        data["losses/clipfrac"] = np.mean(clipfracs)
        data["losses/approx_kl"] = approx_kl.item()
        data["losses/all_loss"] = loss.item()
        data["charts/SPS"] = int(global_step / (time.time() - start_time))

        data["rewards/rewards_mean"] = rewards.mean().item()
        data["rewards/rewards_max"] = rewards.max().item()
        data["rewards/rewards_min"] = rewards.min().item()
        data["rewards/true_rewards_mean"] = true_rewards.mean().item()
        data["rewards/true_rewards_max"] = true_rewards.max().item()
        data["rewards/true_rewards_min"] = true_rewards.min().item()
        data["rewards/intrinsic_rewards_mean"] = rle_rewards.mean().item()
        data["rewards/intrinsic_rewards_max"] = rle_rewards.max().item()
        data["rewards/intrinsic_rewards_min"] = rle_rewards.min().item()

        data["returns/advantages"] = b_advantages.mean().item()
        data["returns/ext_advantages"] = b_ext_advantages.mean().item()
        data["returns/int_advantages"] = b_int_advantages.mean().item()
        data["returns/ret_ext"] = b_ext_returns.mean().item()
        data["returns/ret_int"] = b_int_returns.mean().item()
        data["returns/values_ext"] = b_ext_values.mean().item()
        data["returns/values_int"] = b_int_values.mean().item()

        data["charts/traj_len"] = np.mean(avg_ep_lens)
        data["charts/max_traj_len"] = np.max(avg_ep_lens, initial=0)
        data["charts/min_traj_len"] = np.min(avg_ep_lens, initial=0)
        data["charts/time_per_it"] = it_end_time - it_start_time
        data["charts/episode_return"] = np.mean(avg_returns)
        data["charts/max_episode_return"] = np.max(avg_returns, initial=0)
        data["charts/min_episode_return"] = np.min(avg_returns, initial=0)
        data["charts/true_episode_return"] = np.mean(avg_true_returns)
        data["charts/max_true_episode_return"] = np.max(avg_true_returns, initial=0)
        data["charts/min_true_episode_return"] = np.min(avg_true_returns, initial=0)

        data["charts/consecutive_successes"] = np.mean(avg_consecutive_successes)
        data["charts/max_consecutive_successes"] = np.max(avg_consecutive_successes, initial=0)
        data["charts/min_consecutive_successes"] = np.min(avg_consecutive_successes, initial=0)

        if args.track:
            wandb.log(data, step=global_step)
