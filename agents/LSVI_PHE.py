from agents.MaxminDQN import *
import random
import math

class LSVI_PHE(MaxminDQN):
  '''
  Implementation of LSVI_PHE with target network and replay buffer.
  We update all Q_nets for every update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    self.noise_std = cfg['agent']['noise_std'] # std of Gaussian noise
    self.l2_lambda = cfg['agent']['lambda'] # noise regularization parameter
    random.seed(self.cfg['seed'])

    self.d = self.action_size * self.state_size
    
    self.phi = np.identity(self.d)

    self.delta_Sigma = np.zeros((self.state_size+8,self.d,self.d))
    
    self.syn_episode = 0
  
  def learn(self):
    mode = 'Train'
    # Use the same batch during training
    batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
    # Update Q network
    for i in range(self.k): 
      self.update_Q_net_index = i
      q, q_target = self.compute_q(batch), self.compute_q_target(batch)
      # Compute L2 noise regularization
      l2_reg = torch.tensor(0.0)

      pytorch_total_params = sum(p.numel() for p in self.Q_net[i].parameters())
      print('tital: ', pytorch_total_params)
      for param in self.Q_net[i].parameters():
        param_noise = to_tensor(torch.randn(param.size())*self.noise_std, device=self.device) #xi
        l2_reg += torch.linalg.norm(param + param_noise)
      # Compute loss
      loss = self.loss(q, q_target) + self.l2_lambda * l2_reg # lambda* norm(weight)
      # Take an optimization step
      self.optimizer[i].zero_grad()
      loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.Q_net[i].parameters(), self.gradient_clip)
      self.optimizer[i].step()
    # Update target network
    if (self.step_count // self.cfg['network_update_steps']) % self.cfg['target_network_update_steps'] == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state[mode], device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    q_values = self.get_action_selection_q_values(state)
    # Alwaya select the best action
    action = np.argmax(q_values)
    
    # if random.random() < 0.01:
    #   action = 0
    return action

  def compute_q_target(self, batch): #  # reward + maxQ
    with torch.no_grad():
      q_max = self.Q_net_target[0](batch.next_state).clone()
      for i in range(1, self.k):
        q = self.Q_net_target[i](batch.next_state)
        q_max = torch.max(q_max, q)
      q_next = q_max.max(1)[0]
      reward_noise = to_tensor(torch.randn(batch.reward.size())*self.noise_std, device=self.device) # epsilon
      q_target = batch.reward + reward_noise + self.discount * q_next * batch.mask
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_max = self.Q_net[0](state)
    for i in range(1, self.k):
      q = self.Q_net[i](state)
      q_max = torch.max(q_max, q)
    q_max = to_numpy(q_max).flatten()
    return q_max

  def save_global_experience(self, h, Sigma, global_replay):

       
    mode = 'Train'
    prediction = {}
    if self.reward[mode] is not None:
     
      # print(self.action[mode])

      row = int((np.sum(self.state[mode])-1)*self.action_size + self.action[mode])
     
      self.delta_Sigma[h] = np.add(self.delta_Sigma[h] , np.outer(self.phi[row,:],self.phi[row,:]))
      Sigma[h] = np.add(Sigma[h] , np.outer(self.phi[row,:],self.phi[row,:]))
         
      prediction['state'] = to_tensor(self.state[mode], self.device)
      prediction['action'] = to_tensor(self.action[mode], self.device)
      prediction['next_state'] = to_tensor(self.next_state[mode], self.device)
      prediction['mask'] = to_tensor(1-self.done[mode], self.device)
      prediction['reward'] = to_tensor(self.reward[mode], self.device)
    self.replay.add(prediction)
    global_replay.add(prediction)

    return global_replay



  def run_parallel_episode(self, mode, Sigma, global_replay, total_episode,render, expon_bool):
    h = 0
    syn_bool = 0
    while not self.done[mode]:
      self.action[mode] = self.get_action(mode)
      if render:
        self.env[mode].render()
      # Take a step
      next_state, self.reward[mode], self.done[mode], _ = self.env[mode].step(self.action[mode])
      self.next_state[mode] = self.state_normalizer(next_state)
      self.reward[mode] = self.reward_normalizer(self.reward[mode])
      self.episode_return[mode] += self.reward[mode]
      self.episode_step_count[mode] += 1
      if mode == 'Train':
        # Save experience
        global_replay =self.save_global_experience(h, Sigma, global_replay)
        
        
        # Update policy
        #print('time to learn')
        if self.time_to_learn():
          self.learn()
       
        # Update target Q network: used only in DQN variants
        self.update_target_net()
       
        self.step_count += 1

        

        if math.log(np.linalg.det(Sigma[h]+self.delta_Sigma[h])/ np.linalg.det(Sigma[h])) > self.threshold/(total_episode- self.syn_episode):
        # if total_episode % 115 == 0: #337 for n = 25
        # if expon_bool:

         
          self.syn_episode = total_episode -1

          
          syn_bool = 1

          for j in range(self.state_size + 8):
            Sigma[j] += self.delta_Sigma[j]
          
          self.delta_Sigma = np.zeros((self.state_size+8,self.d,self.d))

          self.replay = global_replay


        h += 1
        
      # Update state
      self.state[mode] = self.next_state[mode]
      self.original_state = next_state
    # End of one episode
    # self.save_episode_result(mode)
    if mode == 'Train':
      self.parallel_return = self.episode_return[mode]
    # Reset environment
    self.reset_game(mode)
    if mode == 'Train':
      self.episode_count += 1

    return Sigma, global_replay, syn_bool

  