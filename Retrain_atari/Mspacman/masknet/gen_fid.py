import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load("/home/zck7060/Retrain_atari/mspacman/masknet/masknet-mspacman", verbose=1)

env = make_atari_env("MsPacman-v0", n_envs=1)
# eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
#reward_vec, length_vec = evaluate_policy(model, env,  n_eval_episodes=500, return_episode_rewards=True)

base_model = PPO.load("/home/zck7060/Retrain_atari/mspacman/baseline/tmp/best_model/best_model", env=env, verbose=1)

# mean_reward, std_reward = evaluate_policy(base_model, base_model.get_env(), n_eval_episodes=10)
# print(mean_reward)

reward_vec = []


for i in range(500):
    reward = 0
    mask_num = 0
    count = 0
    action_seq = []
    mask_probs = []
    env.seed(i)
    obs = env.reset()


    while True:

        action, _states = model.predict(obs, deterministic=True)
        base_action, _states = base_model.predict(obs, deterministic=True)
        obs, vectorized_env = model.policy.obs_to_tensor(obs)
        mask_dist = model.policy.get_distribution(obs)
        mask_prob = np.exp(mask_dist.log_prob(torch.Tensor([1]).cuda()).detach().cpu().numpy()[0])
        #print(mask_prob)
        mask_probs.append(mask_prob)
        
        if action[0]== 0:
            mask_num += 1
        obs, rewards, dones, infos = env.step(base_action)
        info = infos[0]
        action_seq.append(base_action[0])


        if "episode" in info.keys():
            reward_vec.append(info["episode"]["r"])
            count = info["episode"]["l"]
            break


    eps_len_filename = "./recording/eps_len_" + str(i) + ".out" 
    np.savetxt(eps_len_filename, [count])

    act_seq_filename = "./recording/act_seq_" + str(i) + ".npy" 
    np.save(act_seq_filename, action_seq, allow_pickle=True)

    mask_probs_filename = "./recording/mask_probs_" + str(i) + ".out" 
    np.savetxt(mask_probs_filename, mask_probs)


    print("Test ", i)
    print(info["episode"]["r"])


print("Average reward: ", np.mean(reward_vec))
print("Reward std: ", np.std(reward_vec))


np.savetxt("./recording/reward_record.out", reward_vec)