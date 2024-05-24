import torch
import numpy as np
import os
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from Agents.PPOAgent_mask import PPOAgent_mask
from Agents.PPOAgent import PPOAgent
import random

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(env, input_dims, action_space,
          max_episodes, max_timesteps, K_epochs, eps_clip,
          gamma, lr, betas, start_actions=[]):


    ckpt_mask = 'Models/bline_mask/4000.pth'
    agent_mask = PPOAgent_mask(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions, restore=True, ckpt=ckpt_mask, deterministic=True, training=False)
    ckpt_ppo = 'Models/bline/99800.pth'
    agent_ppo = PPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions, restore=True, ckpt=ckpt_ppo, deterministic=True, training=False)


    time_step = 0
    Rewards = 0

    mask_record = []
    action_record = []
    state = env.reset()

    for t in range(max_timesteps):
        time_step += 1
        action = agent_ppo.get_action(state)
        mask_prob = agent_mask.get_mask_prob(state)[0][0]  #here the [0] denotes the prob of not masking
        mask_record.append(mask_prob.detach().cpu())
        action_record.append(action)
        state, reward, done, _ = env.step(action)

    agent_ppo.end_episode()
    agent_mask.end_episode()


    return action_record, mask_record




def gen_one_traj(seed):
    # set seeds for reproducibility
    set_seed(seed)

    CYBORG = CybORG(PATH, 'sim', agents={
        'Red': B_lineAgent
    })
    env = ChallengeWrapper2(env=CYBORG, agent_name="Blue")
    input_dims = env.observation_space.shape[0]

    action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    action_space += [11, 12, 13, 14]  # analyse user hosts
    action_space += [141, 142, 143, 144]  # restore user hosts
    action_space += [132]  # restore defender
    action_space += [2]  # analyse defender
    action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts

    start_actions = [1004, 1004, 1000] # user 2 decoy * 2, ent0 decoy

    max_episodes = 500
    max_timesteps = 100

    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002


    return train(env, input_dims, action_space,
                max_episodes=max_episodes, max_timesteps=max_timesteps,
                K_epochs=K_epochs,
                eps_clip=eps_clip, gamma=gamma, lr=lr,
                betas=[0.9, 0.990], start_actions=start_actions)