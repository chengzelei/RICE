# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

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
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, start_actions=[]):


    ckpt_ppo = '/home/jys3649/projects/xai_masknet/cyborg-cage-2/Models/bline/99800.pth'
    agent_ppo = PPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions, restore=True, ckpt=ckpt_ppo, deterministic=True, training=False)


    running_reward, time_step = 0, 0
    reward_log = []
    action_log = []
    value_highlihgts_log = []
    value_trust_log = []
    value_max_log = []
    Rewards = 0
    for i_episode in range(max_episodes):
        value_highlights_record = []
        value_trust_record = []
        value_max_record = []
        action_record = []
        set_seed(i_episode)
        state = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            time_step += 1
            action = agent_ppo.get_action(state)
            values = agent_ppo.get_action_prob(state)[0].detach().cpu().numpy()
            value_trust_record.append(np.max(values) - np.mean(values))
            value_highlights_record.append(np.max(values) - np.min(values))
            value_max_record.append(np.max(values))
            action_record.append(action)
            state, reward, done, _ = env.step(action)


            total_reward += reward

        agent_ppo.end_episode()
        print("Episode {}, Reward {}".format(i_episode, total_reward))
        Rewards += total_reward
        reward_log.append(total_reward)
        value_highlihgts_log.append(value_highlights_record)
        value_trust_log.append(value_trust_record)
        value_max_log.append(value_max_record)
        action_log.append(action_record)

    path = './recordings/'
    print("Average Reward {}".format(Rewards/max_episodes))
    reward_log = np.array(reward_log)
    value_highlihgts_log = np.array(value_highlihgts_log)
    value_trust_log = np.array(value_trust_log)
    action_log = np.array(action_log, dtype=np.int32)
    np.savetxt(path + "reward_log.txt", reward_log)
    np.savetxt(path + "highlights_prob_log.txt", value_highlihgts_log)
    np.savetxt(path + "trust_prob_log.txt", value_trust_log)
    np.savetxt(path + "max_prob_log.txt", value_max_log)
    np.savetxt(path + "action_log.txt", action_log)




if __name__ == '__main__':

    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # change checkpoint directory
    folder = 'bline_mask'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

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

    print_interval = 50
    save_interval = 200
    max_episodes = 500
    max_timesteps = 100
    # 200 episodes for buffer
    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002


    train(env, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], ckpt_folder=ckpt_folder,
              print_interval=print_interval, save_interval=save_interval, start_actions=start_actions)