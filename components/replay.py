import torch
import numpy as np
from collections import namedtuple
from Utils.helper import to_tensor
import random
class SumTree:
  def __init__(self, size):
    self.nodes = [0] * (2 * int(size) - 1)
    self.data = [None] * int(size)

    self.size = int(size)
    self.count = 0
    self.real_size = 0

  @property
  def total(self):
      return self.nodes[0]

  def update(self, data_idx, value):
      idx = data_idx + self.size - 1  # child index in tree array
      change = value - self.nodes[idx]

      self.nodes[idx] = value

      parent = (idx - 1) // 2
      while parent >= 0:
          self.nodes[parent] += change
          parent = (parent - 1) // 2

  def add(self, value, data):
      self.data[self.count] = data
      self.update(self.count, value)

      self.count = (self.count + 1) % self.size
      self.real_size = min(self.size, self.real_size + 1)

  def get(self, cumsum):
      assert cumsum <= self.total

      idx = 0
      while 2 * idx + 1 < len(self.nodes):
          left, right = 2*idx + 1, 2*idx + 2

          if cumsum <= self.nodes[left]:
              idx = left
          else:
              idx = right
              cumsum = cumsum - self.nodes[left]

      data_idx = idx - self.size + 1

      return data_idx, self.nodes[idx], self.data[data_idx]

  def __repr__(self):
      return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
class InfiniteReplay(object):
  '''
  Infinite replay buffer to store experiences
  '''
  def __init__(self, keys=None):
    if keys is None:
      keys = ['action', 'reward', 'mask']
    self.keys = keys
    self.clear()

  def add(self, data):
    for k, v in data.items():
      if k not in self.keys:
        raise RuntimeError('Undefined key')
      getattr(self, k).append(v)

  def placeholder(self, data_size):
    for k in self.keys:
      v = getattr(self, k)
      if len(v) == 0:
        setattr(self, k, [None] * data_size)

  def clear(self):
    for key in self.keys:
      setattr(self, key, [])

  def get(self, keys, data_size):
    data = [getattr(self, k)[:data_size] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))





class Prioritized_FiniteReplay(object):# eps=1e-2, alpha=0.1, beta=0.1
  '''
  Finite replay buffer to store experiences: FIFO (first in, firt out)
  '''
  def __init__(self, memory_size, keys=None,  eps=1e-2, alpha=0.05, beta=0.1):
    if keys is None:
      keys = ['action', 'reward', 'mask']
    self.keys = keys
    self.memory_size = int(memory_size)

    self.tree = SumTree(size=memory_size)
    self.eps = eps  # minimal priority, prevents zero probabilities
    self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
    self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
    self.max_priority = eps  # priority for new samples, init as eps
    self.real_size = 0
    self.clear()
    
    

  def clear(self):
    self.pos = 0
    self.full = False
    for key in self.keys:
      setattr(self, key, [None] * self.memory_size)
  def update_priorities(self, data_idxs, priorities):
    if isinstance(priorities, torch.Tensor):
        priorities = priorities.detach().cpu().numpy()

    for data_idx, priority in zip(data_idxs, priorities):
        # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
        # where eps is a small positive constant that prevents the edge-case of transitions not being
        # revisited once their error is zero. (Section 3.3)
        priority = (priority + self.eps) ** self.alpha

        self.tree.update(data_idx, priority)
        self.max_priority = max(self.max_priority, priority)


  def add(self, data):
    for k, v in data.items():
      if k not in self.keys:
        raise RuntimeError('Undefined key')
      getattr(self, k)[self.pos] = v
    self.tree.add(self.max_priority, self.pos)
    self.pos = (self.pos + 1) % self.memory_size
    if self.pos == 0:
      self.full = True
    self.real_size = min(self.memory_size, self.real_size + 1)
    
    # store transition index with maximum priority in sum tree
    

  def get(self, keys, data_size, detach=False):
    # Get first several samples (without replacement)
    data_size = min(self.size(), data_size) 
    data = [getattr(self, k)[:data_size] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))

  def sample(self, keys, batch_size, detach=False):
    # Sampling with replacement
    idxs = np.random.randint(0, self.size(), size=batch_size)

    idxs, tree_idxs = [], []
    priorities = torch.empty(batch_size, 1, dtype=torch.float)
    segment = self.tree.total / batch_size


    for i in range(batch_size):
        a, b = segment * i, segment * (i + 1)

        cumsum = random.uniform(a, b)
        # sample_idx is a sample index in buffer, needed further to sample actual transitions
        # tree_idx is a index of a sample in the tree, needed further to update priorities
        tree_idx, priority, sample_idx = self.tree.get(cumsum)

        priorities[i] = priority
        tree_idxs.append(tree_idx)
        idxs.append(sample_idx)

    # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
    # where p_i > 0 is the priority of transition i. (Section 3.3)
    probs = priorities / self.tree.total
    weights = (self.real_size * probs) ** -self.beta
    weights = weights / weights.max()

    data = [[getattr(self, k)[idx] for idx in idxs] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data)), weights, tree_idxs

  def is_empty(self):
    if self.pos == 0 and not self.full:
      return True
    else:
      return False
  
  def is_full(self):
    return self.full

  def size(self):
    if self.full:
      return self.memory_size
    else:
      return self.pos

class FiniteReplay(object):
  '''
  Finite replay buffer to store experiences: FIFO (first in, firt out)
  '''
  def __init__(self, memory_size, keys=None):
    if keys is None:
      keys = ['action', 'reward', 'mask']
    self.keys = keys
    self.memory_size = int(memory_size)
    self.clear()
    

  def clear(self):
    self.pos = 0
    self.full = False
    for key in self.keys:
      setattr(self, key, [None] * self.memory_size)

  def add(self, data):
    for k, v in data.items():
      if k not in self.keys:
        raise RuntimeError('Undefined key')
      getattr(self, k)[self.pos] = v
    self.pos = (self.pos + 1) % self.memory_size
    if self.pos == 0:
      self.full = True

  def get(self, keys, data_size, detach=False):
    # Get first several samples (without replacement)
    data_size = min(self.size(), data_size) 
    data = [getattr(self, k)[:data_size] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))

  def sample(self, keys, batch_size, detach=False):
    # Sampling with replacement
    idxs = np.random.randint(0, self.size(), size=batch_size)
    data = [[getattr(self, k)[idx] for idx in idxs] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))

  def is_empty(self):
    if self.pos == 0 and not self.full:
      return True
    else:
      return False
  
  def is_full(self):
    return self.full

  def size(self):
    if self.full:
      return self.memory_size
    else:
      return self.pos


class ContinousUniformSampler(object):
  '''
  A uniform sampler for continous space
  '''
  def __init__(self, shape, normalizer, device):
    self.shape = shape
    self.normalizer = normalizer
    self.device = device
    self.reset()
  
  def reset(self):
    self.low = np.inf * np.ones(self.shape)
    self.high = -np.inf * np.ones(self.shape)
  
  def update_bound(self, data):
    self.low = np.minimum(self.low, data)
    self.high = np.maximum(self.high, data)

  def sample(self, batch_size):
    data = np.random.uniform(low=self.low, high=self.high, size=tuple([batch_size]+list(self.shape)))
    data = to_tensor(self.normalizer(data), self.device)
    return data


class DiscreteUniformSampler(ContinousUniformSampler):
  '''
  A uniform sampler for discrete space
  '''
  def __init__(self, shape, normalizer, device):
    super().__init__(shape, normalizer, device)

  def sample(self, batch_size):
    data = np.random.randint(low=np.rint(self.low), high=np.rint(self.high)+1, size=tuple([batch_size]+list(self.shape)))
    data = to_tensor(self.normalizer(data), self.device)
    return data