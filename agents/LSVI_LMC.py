from agents.DQN import *
from collections import Counter


class LSVI_LMC(DQN):
  '''
  Implementation of Bootstrapped DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    self.k = cfg['agent']['update_num'] # number of update iteration
    super().__init__(cfg)


    self.d = self.action_size * self.state_size
    
    self.phi = np.identity(self.d)

    self.delta_Sigma = np.zeros((self.state_size+8,self.d,self.d))
    
    self.syn_episode = 0


  def run_episode(self, mode, render):
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
        self.save_experience()
        # Update policy
        #print('time to learn')
        if self.time_to_learn():

          for iters in range(self.k):
            self.learn()
       
        # Update target Q network: used only in DQN variants
        self.update_target_net()
       
        self.step_count += 1
      # Update state
      self.state[mode] = self.next_state[mode]
      self.original_state = next_state
    # End of one episode
    self.save_episode_result(mode)
    # Reset environment
    self.reset_game(mode)
    
    if mode == 'Train':
      self.episode_count += 1
  
  def save_global_experience(self, h, Sigma, global_replay):

       
    mode = 'Train'
    prediction = {}
    if self.reward[mode] is not None:
     
      # print(self.action[mode])

      row = int((np.sum(self.state[mode])-1)*self.action_size + self.action[mode])
     
      self.delta_Sigma[h] = np.add(self.delta_Sigma[h] , np.outer(self.phi[row,:],self.phi[row,:]))
      # Sigma[h] = np.add(Sigma[h] , np.outer(self.phi[row,:],self.phi[row,:]))
         
      prediction['state'] = to_tensor(self.state[mode], self.device)
      prediction['action'] = to_tensor(self.action[mode], self.device)
      prediction['next_state'] = to_tensor(self.next_state[mode], self.device)
      prediction['mask'] = to_tensor(1-self.done[mode], self.device)
      prediction['reward'] = to_tensor(self.reward[mode], self.device)
    self.replay.add(prediction)
    global_replay.add(prediction)

    return global_replay

  def run_parallel_episode(self, mode, Sigma, global_replay, total_episode, render, expon_bool):
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
        if self.time_to_learn():

          for iters in range(self.k):
            self.learn()
       
        # Update target Q network: used only in DQN variants
        self.update_target_net()
       
        self.step_count += 1
        # print((Sigma[h], self.delta_Sigma[h]), np.linalg.det(Sigma[h]))
        # print( math.log(np.linalg.det(Sigma[h]+self.delta_Sigma[h])/ np.linalg.det(Sigma[h])))
        # print(math.log(np.linalg.det(Sigma[h]+self.delta_Sigma[h])/ np.linalg.det(Sigma[h])),  self.threshold/(total_episode- self.syn_episode))
        if math.log(np.linalg.det(Sigma[h]+self.delta_Sigma[h])/ np.linalg.det(Sigma[h])) > self.threshold/(total_episode- self.syn_episode):
        #if total_episode % 148 == 0: # 115 is for n=10, 148 is for n=25
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