from agents.VanillaDQN import *


class MaxminDQN(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer

  We can update all Q_nets for every update. However, this makes training really slow.
  Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    # Create k different: Q value network, Target Q value network and Optimizer
    self.Q_net = [None] * self.k
    self.Q_net_target = [None] * self.k
    self.optimizer = [None] * self.k
    for i in range(self.k):
      self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(), **cfg['optimizer']['kwargs'])
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
      self.Q_net_target[i].eval()

  def learn(self):
    # Choose a Q_net to udpate
    self.update_Q_net_index = np.random.choice(list(range(self.k)))
    super().learn()

  def update_target_net(self):
    if self.step_count % self.cfg['target_network_update_steps'] == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())

  def compute_q_target(self, batch):
    with torch.no_grad():
      q_min = self.Q_net_target[0](batch.next_state).clone()
      for i in range(1, self.k):
        q = self.Q_net_target[i](batch.next_state)
        q_min = torch.min(q_min, q)
      q_next = q_min.max(1)[0]
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_min = self.Q_net[0](state)
    for i in range(1, self.k):
      q = self.Q_net[i](state)
      q_min = torch.min(q_min, q)
    q_min = to_numpy(q_min).flatten()
    return q_min

  

  ################# phe
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
  

   
  






