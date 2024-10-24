# RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation

This repo contains the code for the paper ["RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation"](https://proceedings.mlr.press/v235/cheng24j.html)

Paper citation:
```
@inproceedings{cheng2024rice,
title={RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation},
author={Cheng, Zelei and Wu, Xian and Yu, Jiahao and Yang, Sabrina and Wang, Gang and Xing, Xinyu},
booktitle={Proc. of ICML},
year={2024}
```

## Requirement
The codebase is written with ```python3.7``` and ```Pytorch```. We provide a `requirements.txt` for your reference. If you run errors in some programs, install the missing lib via pip install as the error report. 

## Introduction
### Basics
- We implement our methods in four dense/sparse mujoco games and four real-world applications.
  
- Note that our implementation of the explanation method is based on [StateMask](https://github.com/nuwuxian/RL-state_mask).

- For each game, we provide code for training a mask network and refining the target agent with an explanation.
