import pickle
import os
from tqdm import tqdm

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

import gym
from gym.wrappers import FrameStack

import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import torch.nn.functional as F
import random



import pandas as pd
env_test = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
print(env_test.observation_space.shape)
print(env_test.action_space.n)
import math
import torch
from torch import Tensor
from typing import List, Optional

from torch.optim.sgd import *
from torch.optim.adam import *
from torch.optim.rmsprop import *


class aSGLD(Adam):
  """
  Implementation of Adam SGLD based on: http://arxiv.org/abs/2009.09535
  Built on PyTorch Adam implementation.
  Note that there is no bias correction in the original description of Adam SGLD.
  """
  def __init__(
    self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
    weight_decay=0, amsgrad=False,
    noise_scale=0.01, a=1.0
  ):
    defaults = dict(
      lr=lr, betas=betas, eps=eps, 
      weight_decay=weight_decay, amsgrad=amsgrad
    )
    super(aSGLD, self).__init__(params, **defaults)
    self.noise_scale = noise_scale
    self.a = a

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    #self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      max_exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is not None:
          params_with_grad.append(p)
          if p.grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
          grads.append(p.grad)

          state = self.state[p]
          # Lazy state initialization
          if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group['amsgrad']:
              # Maintains max of all exp. moving avg. of sq. grad. values
              state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

          exp_avgs.append(state['exp_avg'])
          exp_avg_sqs.append(state['exp_avg_sq'])

          if group['amsgrad']:
            max_exp_avg_sqs.append(state['max_exp_avg_sq'])
          
          # update the steps for each param group update
          state['step'] += 1
          # record the step after step update
          state_steps.append(state['step'])

      adam_sgld(
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=group['amsgrad'],
        beta1=beta1,
        beta2=beta2,
        lr=group['lr'],
        weight_decay=group['weight_decay'],
        eps=group['eps'],
        noise_scale=self.noise_scale,
        a=self.a
      )
    return loss


def adam_sgld(
  params: List[Tensor],
  grads: List[Tensor],
  exp_avgs: List[Tensor],
  exp_avg_sqs: List[Tensor],
  max_exp_avg_sqs: List[Tensor],
  state_steps: List[int],
  *,
  amsgrad: bool,
  beta1: float,
  beta2: float,
  lr: float,
  weight_decay: float,
  eps: float,
  noise_scale: float,
  a: float
):
  """Functional API that performs Adam SGLD algorithm computation.
  See :class:`~torch.optim.Adam` for details.
  """
  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step = state_steps[i]

    if weight_decay != 0:
      grad = grad.add(param, alpha=weight_decay)

    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
    if amsgrad:
      # Maintains the maximum of all 2nd moment running avg. till now
      torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
      # Use the max. for normalizing running avg. of gradient
      denom = max_exp_avg_sqs[i].sqrt().add_(eps)
    else:
      denom = exp_avg_sq.sqrt().add_(eps)
    
    # Add pure gradient
    param.add_(grad, alpha=-lr)
    # Add the adaptive bias term
    am = a * exp_avg
    param.addcdiv_(am, denom, value=-lr)
    # Add noise
    grad_perturb = torch.normal(0, 1, size=param.shape, dtype=param.dtype, device=param.device)
    param.add_(noise_scale * math.sqrt(2.0*lr) * grad_perturb)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                         shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                         shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

def create_mario_env(env_name):
    env = gym_super_mario_bros.make(env_name)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env #JoypadSpace(env, SIMPLE_MOVEMENT)



import numpy as np

class ReplayBuffer:
    def __init__(self, state_shape, action_space, batch_size=32, max_size=10000,
                 load=False, path=None):
        self.path = path + 'buffer/'
        self.max_size = max_size
        self.batch_size = batch_size

        if load:
            self.load()
        else:
            self.next = 0
            self.size = 0

            self.states = torch.empty((max_size, *state_shape))
            self.actions = torch.empty((max_size, 1), dtype=torch.int64)
            self.rewards = torch.empty((max_size, 1))
            self.states_p = torch.empty((max_size, *state_shape))
            self.is_terminals = torch.empty((max_size, 1), dtype=torch.float)


    def __len__(self): return self.size
    

    def store(self, state, action, reward, state_p, is_terminal):
        state = state.__array__()
        state_p = state_p.__array__()

        self.states[self.next] = torch.tensor(state)
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.states_p[self.next] = torch.tensor(state_p)
        self.is_terminals[self.next] = is_terminal

        self.size = min(self.size + 1, self.max_size)
        self.next = (self.next + 1) % self.max_size


    def sample(self):
        indices = np.random.choice(self.size, size=self.batch_size, 
                                   replace=False)
        return self.states[indices], \
            self.actions[indices], \
            self.rewards[indices], \
            self.states_p[indices], \
            self.is_terminals[indices]


    def clear(self):
        self.next = 0
        self.size = 0
        self.states = torch.empty_like(self.states)
        self.actions = torch.empty_like(self.actions)
        self.rewards = torch.empty_like(self.rewards)
        self.states_p = torch.empty_like(self.states_p)
        self.is_terminals = torch.empty_like(self.is_terminals)


    def load(self):
        with open(self.path + "next.pkl", 'rb') as f:
            self.next = pickle.load(f)
        with open(self.path + "size.pkl", 'rb') as f:
            self.size = pickle.load(f)
        self.states = torch.load(self.path + "states.pt")
        self.actions = torch.load(self.path + "actions.pt")
        self.rewards = torch.load(self.path + "rewards.pt")
        self.states_p = torch.load(self.path + "states_p.pt")
        self.is_terminals = torch.load(self.path + "is_terminals.pt")


    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + "next.pkl", "wb") as f:
            pickle.dump(self.next, f)
        with open(self.path + "size.pkl", "wb") as f:
            pickle.dump(self.size, f)
        torch.save(self.states, self.path + "states.pt")
        torch.save(self.actions, self.path + "actions.pt")
        torch.save(self.rewards, self.path + "rewards.pt")
        torch.save(self.states_p, self.path + "states_p.pt")
        torch.save(self.is_terminals, self.path + "is_terminals.pt")
# Adapted from https://github.com/Kaixhin/Rainbow/blob/master/model.py
class NoisyLinear(nn.Module):
  '''
  Noisy linear layer with Factorised Gaussian noise
  '''
  def __init__(self, in_features, out_features, std_init=0.4):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.Tensor(out_features))
    self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
    self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
    self.register_buffer('bias_epsilon', torch.Tensor(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class QNetwork(torch.nn.Module):
    def __init__(self, input_shape, actions_size, 
                optimizer=torch.optim.Adam, learning_rate=0.00025, noise = False, algo = None):
        super().__init__()
        self.noise = noise
        self.personalized = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
        )
        if self.noise:
            self.shared = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            NoisyLinear(3136, 512),
            torch.nn.ReLU(),
            NoisyLinear(512, actions_size)
            )
        else:
            self.shared = torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(3136, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, actions_size)
            )
        self.algo = algo
        if self.algo == 'lmc':
            self.optimizer = aSGLD(self.parameters(), lr=learning_rate)
        
        else:
            self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()


    def format_(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        return states


    def forward(self, x):
        states = self.format_(x)
        out = self.personalized(states)
        out = self.shared(out)
        return out


    
    def update_netowrk(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def reset_noise(self):
        for layer in self.shared:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

   

class Agent():
    def __init__(self, id, env_name, env_fn, Qnet=QNetwork, buffer=ReplayBuffer,
                 max_epsilon=1, min_epsilon=0.05, epsilon_decay=0.99, gamma=0.9,
                 target_update_rate=2000, min_buffer=100, 
                 load=False, path=None,noise = False, algo = 'dqn') -> None:
        self.id = id
        self.path = path + str(id) + "/"
        self.noise = noise
        self.algo = algo

        self.env = env_fn(env_name)
       
        self.env_fn = env_fn
        self.n_actions = 7 #self.env.action_space.n
        self.state_shape = self.env.observation_space.shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.min_buffer = min_buffer
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_rate = target_update_rate
        self.buffer = buffer(self.state_shape, self.n_actions,
                             load=load, path=self.path)
        

        if self.algo == 'boot' or self.algo == 'phe':
            self.online_nets = []
            self.target_nets = []
            for i in range(4):
                self.online_nets.append(Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device))
                self.target_nets.append(Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device))
        
        
        else:
            self.online_net = Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device)
            self.target_net = Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device)

        if load:
            self.load()
        else:
            if self.algo == 'boot' or self.algo == 'phe':
                self.update_targets_network()
            else:
                self.update_target_network()
            self.epsilon = max_epsilon
            self.step_count = 0
            self.episode_count = 0
            self.rewards = []

    
    def load(self):
        with open(self.path + "step_count.pkl", 'rb') as f:
            self.step_count = pickle.load(f)
        with open(self.path + "episode_count.pkl", 'rb') as f:
            self.episode_count = pickle.load(f)
        with open(self.path + "rewards.pkl", 'rb') as f:
            self.rewards = pickle.load(f)
        with open(self.path + "epsilon.pkl", 'rb') as f:
            self.epsilon = pickle.load(f)
        self.online_net.load_state_dict(torch.load(self.path + "online_net.pt", 
                                                   map_location=torch.device(self.device)))
        self.target_net.load_state_dict(torch.load(self.path + "target_net.pt", 
                                                   map_location=torch.device(self.device)))

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.buffer.save()
        with open(self.path + "step_count.pkl", "wb") as f:
            pickle.dump(self.step_count, f)
        with open(self.path + "episode_count.pkl", "wb") as f:
            pickle.dump(self.episode_count, f)
        with open(self.path + "rewards.pkl", "wb") as f:
            pickle.dump(self.rewards, f)
        with open(self.path + "epsilon.pkl", "wb") as f:
            pickle.dump(self.epsilon, f)
        torch.save(self.online_net.state_dict(), self.path +  "online_net.pt")
        torch.save(self.target_net.state_dict(), self.path +  "target_net.pt")



    def train(self, n_episodes):
        for i in tqdm(range(n_episodes)):
            episode_reward = 0
            state = self.env.reset()

            while True:
                self.step_count += 1
                
                action = self.epsilonGreedyPolicy(state)
                state_p, reward, done, info = self.env.step(action)

                episode_reward += reward

                is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                is_failure = done and not is_truncated
                self.buffer.store(state, action, reward, state_p, float(is_failure))

                if len(self.buffer) >= self.min_buffer:
                    if self.algo == 'lmc':
                        for i in range(4):
                            self.update()

                    else:
                        self.update()
                    if self.step_count % self.target_update_rate == 0:
                        if self.algo == 'boot' or self.algo =='phe':
                            self.update_targets_network()
                        else:
                            self.update_target_network()

                state = state_p
                if done:
                    self.episode_count += 1
                    self.rewards.append(episode_reward)
                    break
        print('info: , ', info)

        print("Agent-{} Episode {} Step {} score = {}, average score = {}"\
                .format(self.id, self.episode_count, self.step_count, self.rewards[-1], np.mean(self.rewards)))


    def get_score(self):
        # return np.mean(self.rewards[-5:])
        return 1


    def update(self):
        states, actions, rewards, states_p, is_terminals = self.buffer.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        states_p = states_p.to(self.device)
        is_terminals = is_terminals.to(self.device)

        if self.noise:
            self.online_net.reset_noise()
            self.target_net.reset_noise()
        

        if self.algo == "phe":
            noise_std = 1e-2
            l2_lambda = 1e-3
            for i in range(4):
                with torch.no_grad():
                    update_Q_net_index = i
                    td_estimate = self.online_nets[update_Q_net_index](states).gather(1, actions)
                q_max = self.target_nets[0](states_p).clone()
                for i in range(4):
                    q = self.target_nets[i](states_p)
                    q_states_p = torch.max(q_max, q)
                q_state_p_action_p = q_states_p.max(1)[0]
                
                reward_noise = torch.as_tensor(torch.randn(self.buffer.batch_size)*noise_std, device=self.device,  dtype=torch.float32) 
                td_target = rewards + reward_noise + (1-is_terminals) * self.gamma * q_state_p_action_p

                # Compute L2 noise regularization
                l2_reg = torch.tensor(0.0).to(self.device)
                for param in self.online_nets[i].parameters():
                    param_noise = torch.as_tensor(torch.randn(param.size())*noise_std, device=self.device,  dtype=torch.float32) #xi
                    l2_reg += torch.linalg.norm(param + param_noise)
                    # Compute loss
                    
                loss = self.online_nets[i].loss_fn(td_estimate, td_target) + l2_lambda * l2_reg

                # Take an optimization step
                self.online_nets[i].optimizer.zero_grad()
                loss.backward()
               
                self.online_nets[i].optimizer.step()


        else:
            if self.algo == 'boot':
            
                td_estimates = []
                for i in range(4):
                    q = self.online_nets[i](states).gather(1, actions).squeeze()
                    td_estimates.append(q)
            
            else:
                td_estimate = self.online_net(states).gather(1, actions)

        
     
            ### calculate tq_target

            if self.algo == 'dqn':
                with torch.no_grad():
                    q_states_p = self.target_net(states_p)
                q_state_p_action_p = q_states_p.max(1)[0]
                td_target = rewards + (1-is_terminals) * self.gamma * q_state_p_action_p
        
       


            elif self.algo == 'boot':
                
                td_targets = []
                
                for i in range(4):
                    with torch.no_grad():
                        q_states_p = self.target_nets[i](states_p)
                    q_state_p_action_p = q_states_p.max(1)[0]
                    td_target = rewards + (1-is_terminals) * self.gamma * q_state_p_action_p
                    td_targets.append(td_target)


            else:
                with torch.no_grad():
                    q_states_p = self.target_net(states_p)
                actions_p = self.online_net(states).argmax(axis=1, keepdim=True)
                q_state_p_action_p = q_states_p.gather(1, actions_p)
                td_target = rewards + (1-is_terminals) * self.gamma * q_state_p_action_p





            if self.algo == 'boot':
                loss = 0
                for i in range(4):
                    loss += self.online_nets[i].loss_fn(td_estimates[i], td_targets[i])
                loss /= 4

                for i in range(4):
                    self.online_nets[i].optimizer.zero_grad()
                loss.backward()
                for i in range(4):
                    self.online_nets[i].optimizer.step()

    

            else:
                self.online_net.update_netowrk(td_estimate, td_target)
        self.update_epsilon()





    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)


    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def update_targets_network(self):
      
        for i in range(4):
            self.target_nets[i].load_state_dict(self.online_nets[i].state_dict())


    def epsilonGreedyPolicy(self, state):

        if self.noise:
            
            self.online_net.reset_noise()

           
            state = state.__array__()
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.online_net(state).argmax().item()

    
        else:
            if np.random.rand() < self.epsilon:
                
                action = np.random.randint(self.n_actions)
            else:
                state = state.__array__()
                state = torch.tensor(state).unsqueeze(0).to(self.device)

                if self.algo == 'boot':
                    head_idx = random.randrange(4)
                    with torch.no_grad():
                        action = self.online_nets[head_idx](state).argmax().item()
                elif self.algo == 'phe':
                    with torch.no_grad():
                        q_max = self.online_nets[0](state)
                        select_net_id = 0
                        for i in range(1, 4):
                            q = self.online_nets[i](state)
                            q_max = torch.max(q_max, q)
                            # if q_max.all() == q.all():
                            #     select_net_id = i
                        # action = self.online_nets[select_net_id](state).argmax().item()
                        action = q_max.argmax().item()
                else:
                    with torch.no_grad():
                        action = self.online_net(state).argmax().item()
   
        return action






class Mario(Agent):
    def __init__(self, env_names, env_fn, Qnet=QNetwork, load=False, path=None, noise = False, algo = 'dqn') -> None:
        print(path)
        self.path = path + "global/"
        self.envs = []
        for name in env_names:
            self.envs.append(env_fn(name))
        self.n_actions = self.envs[0].action_space.n
        self.state_shape = self.envs[0].observation_space.shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.algo = algo
        self.noise = noise
      

        if self.algo == 'boot' or self.algo == 'phe':
            self.online_nets = []
            self.target_nets = []
            for i in range(4):
                self.online_nets.append(Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device))
                self.target_nets.append(Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device))
            # self.online_nets = ([ Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo, algo = self.algo).to(self.device)] for _ in range(4))
            # self.target_nets =  ([ Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo, algo = self.algo).to(self.device)] for _ in range(4))

        # elif self.algo =='phe':
        #     self.online_nets = [None] * 4
        #     self.target_nets = [None] * 4
        #     for i in range(4):
        #         self.online_nets [i] = Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo, algo = self.algo).to(self.device)
        #         self.target_nets [i] = Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo, algo = self.algo).to(self.device)
      
      
  
        
        else:
            self.online_net = Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device)
            self.target_net = Qnet(self.state_shape, self.n_actions, noise = self.noise, algo = self.algo).to(self.device)


        if load:
            self.load()
        else:
            if self.algo == 'boot' or self.algo == 'phe':
                self.update_targets_network()
            else:
                self.update_target_network()



    def load(self):
        self.online_net.load_state_dict(torch.load(self.path + "online_net.pt", 
                                                   map_location=torch.device(self.device)))
        self.target_net.load_state_dict(torch.load(self.path + "target_net.pt", 
                                                   map_location=torch.device(self.device)))


    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(self.online_net.state_dict(), self.path + "online_net.pt")
        torch.save(self.target_net.state_dict(), self.path + "target_net.pt")


    def get_score(self):
        # return np.mean(self.rewards[-5:])
        return 1


    def test(self):
        rewards = np.zeros(len(self.envs))
        for i in range(len(self.envs)):
            r = self.evaluate(i)
            rewards[i] = r
        return rewards


    def evaluate(self, i):
        rewards = 0
        state = self.envs[i].reset()
        while True:
            action = self.greedyPolicy(state)
            state_p, reward, done, _ = self.envs[i].step(action)
            rewards += reward
            if done:
                break
            state = state_p
        return rewards


    def greedyPolicy(self, state):
        with torch.no_grad():
            state = state.__array__()
            state = torch.tensor(state).unsqueeze(0).to(self.device)
            action = self.target_net(state).argmax().item()
        return action
     

class Federator:
    def __init__(self, env_fn, update_rate, path="./Mario/", seed=0,load=False, algo='dqn') -> None:

        self.algo = algo
        if self.algo == 'noise':
            self.noise = True
        else:
            self.noise = False
        self.path = path
        self.envs = [
                'SuperMarioBros-1-1-v0',
                'SuperMarioBros-1-2-v0',
                'SuperMarioBros-1-3-v0',
                'SuperMarioBros-1-4-v0'
        ]
        self.global_agent = Mario(self.envs, env_fn, load=load,path=self.path, noise= self.noise, algo = self.algo )


        self.update_rate = update_rate
        self.n_agents = 4
        self.agents = []
        self.seed = seed
        for i in range(self.n_agents):
            agent = Agent(i, self.envs[i], env_fn, load=load, path=self.path, noise = self.noise, algo = self.algo)
            self.agents.append(agent)

        if load:
            self.load()
        else:

            self.set_local_networks()
            self.rewards = []
        
        self.total_save_rewards = {}


    def load(self):
        with open(self.path + "rewards.pkl", 'rb') as f:
            self.rewards = pickle.load(f)


    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path + "rewards.pkl", "wb") as f:
            pickle.dump(self.rewards, f)
        self.global_agent.save()
        for agent in self.agents:
            agent.save()
        print("All Saved to " + self.path)

    def train(self, n_runs):
        rewards = np.zeros((n_runs, len(self.envs)))
        for i in range(n_runs):
            print("Iteration: {}".format(i+1))
            scores = []
            for agent_id, agent in enumerate(self.agents):
                # print('agent: seed 10 boot', agent_id)
                agent.train(self.update_rate)

                self.total_save_rewards['agent'+str(agent_id)] = agent.rewards
                scores.append(agent.get_score())
            self.aggregate_networks(scores)
            self.set_local_networks()
            # rewards[i] = self.global_agent.test()
            # print(rewards[i])
            # self.save()
            
            if (i % 3 == 0) or i == n_runs-1:
                df = pd.DataFrame(self.total_save_rewards)
                df.to_csv('dqn_agent'+str(agent_id)+'_'+str(self.seed)+'.csv')

                print('save our model')
            
 

    def aggregate_networks(self, scores):
        print('aggregate score: ', scores)
        if self.algo == 'boot':
            for i in range(4):
                sd_online = self.global_agent.online_nets[i].state_dict()
                sd_target = self.global_agent.target_nets[i].state_dict()
                
                online_dicts = []
                target_dicts = []
                for agent in self.agents:
                    online_dicts.append(agent.online_nets[i].state_dict())
                    target_dicts.append(agent.target_nets[i].state_dict())

                for key in sd_online:
                    sd_online[key] = torch.zeros_like(sd_online[key])
                    for i, dict in enumerate(online_dicts):
                        sd_online[key] += scores[i] * dict[key]
                    sd_online[key] /= sum(scores)

                for key in sd_target:
                    sd_target[key] = torch.zeros_like(sd_target[key])
                    for i, dict in enumerate(target_dicts):
                        sd_target[key] += scores[i] * dict[key]
                    sd_target[key] /= sum(scores)
                self.global_agent.online_nets[i].load_state_dict(sd_online)
                self.global_agent.target_nets[i].load_state_dict(sd_target)
            

        
        else:
            sd_online = self.global_agent.online_net.state_dict()
            sd_target = self.global_agent.target_net.state_dict()

            online_dicts = []
            target_dicts = []
            for agent in self.agents:
                online_dicts.append(agent.online_net.state_dict())
                target_dicts.append(agent.target_net.state_dict())

            for key in sd_online:
                sd_online[key] = torch.zeros_like(sd_online[key])
                for i, dict in enumerate(online_dicts):
                    sd_online[key] += scores[i] * dict[key]
                sd_online[key] /= sum(scores)

            for key in sd_target:
                sd_target[key] = torch.zeros_like(sd_target[key])
                for i, dict in enumerate(target_dicts):
                    sd_target[key] += scores[i] * dict[key]
                sd_target[key] /= sum(scores)

            
                
                    
           
            self.global_agent.online_net.load_state_dict(sd_online)
            self.global_agent.target_net.load_state_dict(sd_target)


    def set_local_networks(self):
        if self.algo == 'boot' or self.algo == 'phe':
            for agent in self.agents:
                for i in range(4):
                    agent.online_nets[i].load_state_dict(
                        self.global_agent.online_nets[i].state_dict())
                    agent.target_nets[i].load_state_dict(
                        self.global_agent.target_nets[i].state_dict())

            
        else:
            for agent in self.agents:
                agent.online_net.load_state_dict(
                    self.global_agent.online_net.state_dict())
                agent.target_net.load_state_dict(
                    self.global_agent.target_net.state_dict())

seeds = [20]



for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    agent = Federator(create_mario_env, 200, "./Mario/", seed, load=False, algo = 'dqn') #200
    # print('rewards: ', agent.rewards)
    agent.train(30)

print('finish')
