import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

model = PPO.load("/home/zck7060/Retrain_mujoco/reacher/masknet/weak_models/best_model/best_model")

env = DummyVecEnv([lambda: gym.make("Reacher-v2")])
env = VecMonitor(env, "weak_models/")
# eval_env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
#reward_vec, length_vec = evaluate_policy(model, env,  n_eval_episodes=500, return_episode_rewards=True)

base_model = PPO.load("/home/zck7060/Retrain_mujoco/reacher/baseline/weak_tmp/best_model/best_model")

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
        
        if action == 0:
            mask_num += 1
        obs, rewards, dones, info = env.step(base_action)

        action_seq.append(base_action)

        reward += rewards[0]

        count += 1
        if dones:
            reward_vec.append(reward)
            eps_len_filename = "./weak_recording/eps_len_" + str(i) + ".out" 
            np.savetxt(eps_len_filename, [count])

            act_seq_filename = "./weak_recording/act_seq_" + str(i) + ".npy" 
            np.save(act_seq_filename, action_seq, allow_pickle=True)

            mask_probs_filename = "./weak_recording/mask_probs_" + str(i) + ".out" 
            np.savetxt(mask_probs_filename, mask_probs)
            break

    print("Test ", i)
    print("reward: ", reward)


print("Average reward: ", np.mean(reward_vec))
print("Reward std: ", np.std(reward_vec))


np.savetxt("./weak_recording/reward_record.out", reward_vec)