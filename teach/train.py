import random

from buffer import HindsightReplayBuffer
import utils

import time


import gym
import torch


import wandb
from curriculum import build_goal_samplers
from goal_env import CurriculumGoalEnv


class Experiment(object):
    def __init__(
            self,
            # environment
            env_id,
            
            # visual?
            from_images=False,

            # reproducibility
            seed=1,
            freq = 1,

            # env
            fix_goals=False,

            # compute
            device='cuda' if torch.cuda.is_available() else 'cpu',

            # replay buffer
            num_resampled_goals=1,
            capacity=1_000_000,

            # agent
            feature_dim=128,
            hidden_sizes=[512, 512, 512],
            log_std_bounds=[-20, 2],
            discount=0.95,
            init_temperature=0.1,
            lr=0.001,
            actor_update_frequency=1,
            critic_tau=0.005,
            critic_target_update_frequency=1,
            batch_size=128,
        
            # evaluation
            num_eval_episodes=10,
            
            # training
            gradient_steps=1, # better for wall clock time. increase for better performance.
            num_timesteps=20_000, # maximum time steps
            num_seed_steps=1_000, # random actions to improve exploration
            update_after=1_000, # when to start updating (off-policy still learns from seed steps)
            eval_every=20, # episodic frequency for evaluation
            save_every=5_000, # how often to save the experiment progress in time steps
            n_candidates = 1000,
            val_history = 5,
            strategy = 'variance',
            **kwargs, # lazily absorb extra args
        
        ):
        self.observation_key = 'observation'
        self.achieved_goal_key = 'achieved_goal'
        self.desired_goal_key = 'desired_goal'
        # Seed
        utils.set_seed_everywhere(seed)

        # Create env
        self.env_id = env_id
        wandb.init(project= 'teach_' + env_id, name = str(seed)+str(freq))

        self.seed = seed
        self.from_images = from_images
        self.fix_goals = fix_goals
        if 'Maze' in self.env_id:
            register(
            id=self.env_id,
            entry_point='maze:ParticleMazeEnv',
            kwargs={'grid_name': 'a', 'reward_type': 'sparse'},
            max_episode_steps=50,
            )

        self.env = gym.make(self.env_id)
        self.env.reset(seed=self.seed)

        self.eval_env = gym.make(self.env_id)
        self.eval_env.reset(seed=self.seed)

        self.env = CurriculumGoalEnv(self.env, self.seed)
        self.eval_env = CurriculumGoalEnv(self.eval_env, self.seed)
        # Create agen

        self.agent = Agent(
            from_images,
            self.env.observation_space,
            self.env.action_space,
            device=device, 
            feature_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            log_std_bounds=log_std_bounds,
            discount=discount,
            init_temperature=init_temperature,
            lr=lr,
            critic_tau=critic_tau,
            batch_size=batch_size
        )
        curriculum_goals, uniform_goals = build_goal_samplers(self.env_id, self.agent, n_candidates, freq, val_history)
        # create buffer
        self.env.set_goal_sampler(curriculum_goals)
        self.eval_env.set_goal_sampler(uniform_goals)
        # update env to use agent encoder for images if necessary
        if self.from_images:
            self.env.set_agent(self.agent) # set the conv encoder for latent distance rewards

        # Create replay buffer
        self.replay_buffer = HindsightReplayBuffer(
            from_images=from_images,
            env = self.env,
            num_resampled_goals=num_resampled_goals,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            capacity=capacity,
            device=device, 
        )

        self.step = 0
        self.num_eval_episodes = num_eval_episodes
        
        self.gradient_steps = gradient_steps
        self.num_timesteps = num_timesteps
        self.num_seed_steps = num_seed_steps
        self.update_after = update_after
        self.eval_every = eval_every
        self.save_every = save_every
    
    def eval(self):
        average_episode_reward = 0
        average_episode_success = 0
        
        
        for episode in range(self.num_eval_episodes):
            
            obs_dict = self.eval_env.reset()
            obs = obs_dict[self.observation_key]
            obs_g = obs_dict[self.desired_goal_key]
            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                action = self.agent.act(obs, obs_g, sample=False)

                next_obs_dict, reward, terminate, truncate, info = self.eval_env.step(action)
                done = terminate or truncate
                done = float(done)
                episode_reward += reward

                achieved_goal = next_obs_dict[self.achieved_goal_key]

                obs = next_obs_dict[self.observation_key]
                obs_g = next_obs_dict[self.desired_goal_key]
                episode_step += 1
                            
            average_episode_reward += episode_reward/self.num_eval_episodes
            average_episode_success += float(info['is_success'])/self.num_eval_episodes
            
        wandb.log({'Eval_Success_Rate': average_episode_success, 'Eval_Reward_Avg':average_episode_reward})
        
    
    def train(self):
        episode = 0
        
        while self.step < self.num_timesteps:
            
            obs_dict = self.env.reset()
            obs = obs_dict[self.observation_key]
            obs_g = obs_dict[self.desired_goal_key]

            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                if self.step % self.save_every == 0:
                    self.agent.save(f'agent.ckpt')
                    
                if self.step < self.num_seed_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(obs, obs_g, sample=True)

                next_obs_dict, reward, terminate, truncate, info = self.env.step(action)
                done = terminate or truncate

                next_obs = next_obs_dict[self.observation_key]

                # Allow infinite bootstrap:
                # If the episode was cut off due to time limit, consider done to be false
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.env.spec.max_episode_steps else done
                episode_reward += reward

                achieved_goal = next_obs_dict[self.achieved_goal_key]

                self.replay_buffer.add(obs, obs_g, achieved_goal, action, reward, next_obs, done, done_no_max)
                
                if self.step >= self.update_after:
                    for gradient_step in range(self.gradient_steps):
                        self.agent.update(self.replay_buffer, gradient_step)

                obs = next_obs_dict[self.observation_key]
                obs_g = next_obs_dict[self.desired_goal_key]
                episode_step += 1
                self.step += 1

            #     **self.agent.info,

            wandb.log({'Train_Success_Rate': float(info['is_success']), 'train_reward':episode_reward, 'timesteps_this_iter' : episode_step, **self.agent.info})
            if episode % self.eval_every == 0:
                self.eval()
            episode += 1

        # one final test
        self.eval()
