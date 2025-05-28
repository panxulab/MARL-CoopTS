# Randomized Exploration in Cooperative Multi-Agent Reinforcement Learning

### <p align="center">[NeurIPS 2024]</p>

<p align="center">
  <a href="https://hlhsu.github.io/">Hao-Lun Hsu</a><sup>*</sup> ·
  <a href="https://scholar.google.com/citations?user=WluAK5cAAAAJ&hl=zh-CN">Weixin Wang</a><sup>*</sup> ·
  <a href="https://people.duke.edu/~mp275/">Miroslav Pajic</a> ·
  <a href="https://panxulab.github.io/">Pan Xu</a>
</p>
<p align="center">
Duke University (<sup>*</sup>indicates equal contribution)
</p>
Official implementation of the paper "Randomized Exploration in Cooperative Multi-Agent Reinforcement Learning" with both Perturbed-History Exploration and Langevin Monte Carlo Exploration in a multi-agent setting.

## Requirements

Although [Explorer](https://github.com/qlan3/Explorer/tree/master) is for single-agent settings, the installation process is similar.
- Python (>=3.6)
- [PyTorch](https://pytorch.org/)
- Others: Please check `requirements.txt`.

## The dependency tree of agent classes for N-chain problem

    Base Agent
      ├── Vanilla DQN
      |     ├── DQN
      |     |    ├── DDQN
      |     |    ├── NoisyNetDQN
      |     |    ├── BootstrappedDQN
      |     |    └── LSVI-LMC
      |     ├── Maxmin DQN ── LSVI-PHE
     
     




## Experiments

To train different exploration strategies for N-chain, please change the context in the configuration file in the file "configs/nchain.json" with the corresponding hyper-parameters. Specifically, we can change (1) the value of n in "env" for the length of the states, (2) the agent name, and (3) the optimizer (e.g.,"aSGLD" for LSVI-LMC "Adam" for all other methods).
Then we can run an experiment for N-chain problem as
```python main.py ```
To train different exploration strategies for Super Mario Bros task is more straightforward. Please select the method you are going to use in the script of mario_main.py. For example, algo = "dqn". Then we can run an experiment for Super Mario Bros task as 
```python mario_main.py ```



## Citation
```
@inproceedings{hsu2024randomized,
 title = {Randomized Exploration in Cooperative Multi-Agent Reinforcement Learning},
 author = {Hsu, Hao-Lun and Wang, Weixin and Pajic, Miroslav and Xu, Pan},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {74617--74689},
 volume = {37},
 year = {2024}
}
```
