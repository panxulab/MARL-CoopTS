import os
import sys
import copy
import time
import json
import torch
import numpy as np
import pandas as pd
import random
import agents
from components.replay import *
import components.exploration
import math
import pandas as pd
# from utils.helper import *


class Experiment(object):
  '''
  Train the agent to play the game.
  '''
  def __init__(self, cfg):
    self.cfg = copy.deepcopy(cfg)
    if torch.cuda.is_available() and 'cuda' in cfg['device']:
      self.device = cfg['device']
    else:
      self.cfg['device'] = 'cpu'
      self.device = 'cpu'
    self.config_idx = cfg['config_idx']
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    # if self.cfg['generate_random_seed']:
    #   self.cfg['seed'] = np.random.randint(int(1e6))
    self.model_path = self.cfg['model_path']
    self.cfg_path = self.cfg['cfg_path']
    self.save_config()
  def run_parallel(self, num_agent):
    self.start_time = time.time()
    random.seed(self.cfg['seed'])
    np.random.seed(self.cfg['seed'])
    torch.manual_seed(self.cfg['seed'])

    global_replay = getattr(components.replay, self.cfg['memory_type'])(self.cfg['memory_size'], keys=['state', 'action', 'next_state', 'reward', 'mask'])
    self.agents = []

    
    expon_bool = False
    total_rewards = []

    for i in range(num_agent):
      agent = getattr(agents, self.agent_name)(self.cfg)
      agent.env['Train'].seed(self.cfg['seed'])
      agent.reset_game('Train')
      agent.step_count = 0
      agent.episode_count = 0
      

      if i == 0:
        d = agent.d
        Sigma = np.zeros((agent.state_size+8,d,d))+ np.identity(d)
        max_episode = int(self.cfg['train_steps']/(agent.state_size+8))
      agent.threshold = max_episode * math.log(num_agent* max_episode)/num_agent/d
      # print('threshold: ', agent.threshold)


      

      self.agents.append(agent)

    print('max_episode: ', max_episode)
    
    dicts = {}
    total_syn = 0
    total_episode = 1

    base = 1.6 #1.24
    expon_index = 1
    
    for episode in range(max_episode):

      agent_index = episode % num_agent
      # print(expon_index)


      if episode > int( (base)**  expon_index):
        expon_index += 1
        expon_bool = False
      else:
        if episode == int( (base)**  expon_index):
          expon_index += 1
          expon_bool = True
        else:
          expon_bool = False


      Sigma, global_replay, syn_bool = self.agents[agent_index].run_parallel_episode('Train', Sigma, global_replay, total_episode, self.cfg['render'], expon_bool)
      total_rewards.append(self.agents[agent_index].parallel_return)
      total_episode += 1

      total_syn += syn_bool

      if syn_bool == 1:
        # print('num syn: ', total_syn)
        for index in range(num_agent):
          if index != agent_index:
            
            
            self.agents[index].delta_Sigma = np.zeros((agent.state_size+8,d,d))

            self.agents[index].replay = global_replay



    
   
    print('bootstrap number of syn3  n=25: ', total_syn)

    
    

      


  def run(self):
    '''
    Run the game for multiple times
    '''
  
    self.start_time = time.time()
    random.seed(self.cfg['seed'])
    np.random.seed(self.cfg['seed'])
    torch.manual_seed(self.cfg['seed'])
    # set_random_seed(self.cfg['seed'])
    self.agent = getattr(agents, self.agent_name)(self.cfg)
    self.agent.env['Train'].seed(self.cfg['seed'])
    # self.agent.env['Train'].action_space.np_random.seed(self.cfg['seed'])
    # self.agent.env['Test'].seed(self.cfg['seed'])
    # self.agent.env['Test'].action_space.np_random.seed(self.cfg['seed'])
    # Train && Test
    self.agent.run_steps(render=self.cfg['render'])
    # Save model
    # self.save_model()
    self.end_time = time.time()
    # self.agent.logger.info(f'Memory usage: {rss_memory_usage():.2f} MB')
    # self.agent.logger.info(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')
  
  def save_model(self):
    self.agent.save_model(self.model_path)
  
  def load_model(self):
    self.agent.load_model(self.model_path)

  def save_config(self):
    cfg_json = json.dumps(self.cfg, indent=2)
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()
