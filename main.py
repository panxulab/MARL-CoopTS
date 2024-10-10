import os
import sys
import argparse

from Utils.sweeper import Sweeper
from Utils.helper import make_dir
from experiment import Experiment
import time
import numpy as np

def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/nchain.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)



  seeds = [123]

  training_times = []

  # for count in range(1, 211):
  exp_name = 'test'#'nchain75_phe_smalladv_trainadv'
  num_agent = 1
  for seed in seeds:

  
    start_time = time.time()
    cfg = sweeper.generate_config_for_idx(args.config_idx)
    
    # Set config dict default value
    cfg.setdefault('network_update_steps', 1)
    cfg['env'].setdefault('max_episode_steps', -1)
    cfg.setdefault('show_tb', True)
    cfg.setdefault('render', False)
    cfg.setdefault('gradient_clip', -1)
    cfg.setdefault('hidden_act', 'ReLU')
    cfg.setdefault('output_act', 'Linear')

    cfg['seed'] = seed
    

    # Set experiment name and log paths
    # cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
    cfg['exp'] = exp_name
    cfg['config_idx'] = 'LMC_seed_'+str(seed)
    if len(args.slurm_dir) > 0:  
      cfg['logs_dir'] = f"{args.slurm_dir}/{cfg['exp']}/{cfg['config_idx']}/"
      make_dir(cfg['logs_dir'])
    else:
      cfg['logs_dir'] = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
    make_dir(f"./logs/{cfg['exp']}/{cfg['config_idx']}/")
    print(f"./logs/{cfg['exp']}/{cfg['config_idx']}/")
   
    cfg['train_log_path'] = cfg['logs_dir'] + 'result_Train.feather'
    cfg['test_log_path'] = cfg['logs_dir'] + 'result_Test.feather'
    cfg['model_path'] = cfg['logs_dir'] + 'model.pt'
    cfg['cfg_path'] = cfg['logs_dir'] + 'config.json'

    exp = Experiment(cfg)

    exp.run_parallel(num_agent)
    # exp.run()
    print('training time: ', time.time() - start_time)
   
 
    

if __name__=='__main__':
  main(sys.argv)
