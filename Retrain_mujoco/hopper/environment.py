import gym
from gym import Wrapper
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn as nn
import random

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm1d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
           module.bias.data.zero_()

class MLP_Net(nn.Module):
    def __init__(self, input_dim, hiddens):

        super(MLP_Net, self).__init__()

        self.RND = nn.Sequential()

        for i in range(len(hiddens)):
            if i == 0:
                self.RND.add_module('mlp_%d' %i, nn.Linear(input_dim, hiddens[i]))
            else:
                self.RND.add_module('mlp_%d' %i, nn.Linear(hiddens[i-1], hiddens[i]))

            if i != len(hiddens) - 1:
                self.RND.add_module('relu_%d' %i, nn.ReLU())

        self.RND.apply(initialize_weights)

    
    def eval(self, obs, act):
        # Preprocess the observation
        obs = np.asarray(obs)

        # np_act = np.zeros((3,))
        # np_act.fill(act)
        np_act = np.asarray(act)

        obs_act = np.concatenate((obs, np_act))
        obs_act = torch.FloatTensor(obs_act)

        # Expand dim
        obs_act = torch.unsqueeze(obs_act, 0)
        ret = self.RND(obs_act)
        ret = torch.squeeze(ret, 0)
        return ret.detach().numpy()

class training_pool():

    def __init__(self, losing_games_file, winning_games_file, ratio):
        self.total_num = 5000
        self.ratio = ratio
        self.losing_games_idxs = self.extract_idxs(losing_games_file)
        self.winning_games_idxs = self.extract_idxs(winning_games_file)
        self.candidates = self.create_pool()
    
    def extract_idxs(self, filename):
        idxs = np.loadtxt(filename)
        return idxs
    
    def create_pool(self):
        losing_idxs_selected = np.random.choice(self.losing_games_idxs, int(self.total_num * self.ratio))
        winning_idxs_selected = np.random.choice(self.winning_games_idxs, int(self.total_num * (1-self.ratio)))
        pool = np.concatenate((losing_idxs_selected, winning_idxs_selected), axis=None)
        return pool

def compute_bonus(features, inv_cov):
    bonus = np.sqrt(np.sum(np.dot(np.dot(features.T, inv_cov), features)))
    return bonus

def go_inv_cov(env, random_net, feat_sz, lamb):
    losing_games_file = 'losing_game.out'
    winning_games_file = 'winning_game.out'
    train_pool = training_pool(losing_games_file, winning_games_file, 1.0)
    idxs_list = train_pool.candidates
    critical_steps_starts = np.loadtxt("critical_steps_starts.out")
    losing_idx = train_pool.losing_games_idxs

    cov, pair_num = np.zeros((feat_sz, feat_sz)), 0 

    # Compute inv_cov from the losing trajectories
    for i in range(len(losing_idx)):

        if i % 10 == 0:
            print("Episode %d" %(i))

        idx = int(losing_idx[i])
        action_sequence_path = "weak_retrain_data/act_seq_" + str(idx) + ".npy"
        recorded_actions = np.load(action_sequence_path, allow_pickle=True)
        env.seed(idx)
        obs, done = env.reset(), False
        count = 0
        print(idx)
        while True and count < 1000:
            act = recorded_actions[count]
            if count >= critical_steps_starts[idx]:
                feature = random_net.eval(obs, act[0])
                cov += np.outer(feature, feature)
                pair_num += 1
            obs, reward, done, info = env.step(act)
            count += 1
            if done: break

    print('Pair num is %d' % pair_num)
    cov /= pair_num 
    cov = lamb * np.identity(feat_sz) + cov

    
    np.savez('inv_cov.npz', cov=cov)

class RetrainEnv(Wrapper):

    def __init__(self, env, random_net, bonus_scale, cov):

        Wrapper.__init__(self, env)
        self.env = env
        self.random_net = random_net
        self.bonus_scale = bonus_scale
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.counter = 0

        # 0: losing 1: wining 
        self.idx = 0
        losing_games_file = 'losing_game.out'
        winning_games_file = 'winning_game.out'
        self.train_pool = training_pool(losing_games_file, winning_games_file, 0.5)
        self.idxs_list = self.train_pool.candidates
        self.critical_steps_starts = np.loadtxt("critical_steps_starts.out")



    def step(self, action):
            
        # obtain needed information from the environment.
        obs, reward, done, info = self.env.step(action)
        info["true_reward"] = reward
        bonus = 0
        if reward != 0:
            self.flag = True
        if self.idx == 0:
            feature = self.random_net.eval(self.obs, action)
            self.cov = self.update_cov(feature)
            bonus = compute_bonus(feature, self.inv_cov)
            #print('bonus is', bonus * self.bonus_scale)
            reward += self.bonus_scale * bonus
        self.obs = obs
        return obs, reward, done, info



    def reset(self):
        self.inv_cov = np.linalg.inv(self.cov)
        self.counter += 1
        if self.counter % 100 == 0:
            self.bonus_scale *= 0.8
            self.counter = 0
        self.flag = False
        i_episode = int(random.choice(self.idxs_list))
        
        if i_episode in self.train_pool.losing_games_idxs:
            self.idx = 0
        else:
            self.idx = 1
        action_sequence_path = "weak_retrain_data/act_seq_" + str(i_episode) + ".npy"
        recorded_actions = np.load(action_sequence_path, allow_pickle=True)

        self.env.seed(i_episode)
        obs = self.env.reset()

        if np.random.rand() > 0.5:
            count = 0
            while count < self.critical_steps_starts[i_episode]:
                act = recorded_actions[count]
                obs, reward, done, info = self.env.step(act)
                count += 1
        
        self.obs = obs
        return obs
    
    def update_cov(self, feature):
        self.cov += np.outer(feature, feature)
        return self.cov


def make_retrain_env(env_name, random_net, bonus_scale, inv_cov):
    env = gym.make(env_name)
    return RetrainEnv(env, random_net, bonus_scale, inv_cov)

if __name__ == '__main__':

    ENV_ID = "Hopper-v3"
    PATH = 'RND'
    env = gym.make(ENV_ID).env
    input_dim = 11+3
    random_net = MLP_Net(input_dim, [500, 500])
    # save random_net
    torch.save(random_net.state_dict(), PATH)
    go_inv_cov(env, random_net, 500, 0.01)