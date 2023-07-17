import numpy as np
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
import pickle

vec_norm_path = "/home/zck7060/Retrain_mujoco/halfcheetah/baseline/weak_models/best_model/vec_normalize.pkl"
with open(vec_norm_path, "rb") as file_handler:
    vec_normalize = pickle.load(file_handler)

def select_critical_steps (critical=True, ratio=0.3):
    steps_starts = []
    steps_ends = []

    for i_episode in range(500):
        mask_probs_path = "./weak_recording/mask_probs_" + str(i_episode) + ".out"
        mask_probs = np.loadtxt(mask_probs_path)

        confs = mask_probs

        iteration_ends_path = "./weak_recording/eps_len_" + str(i_episode) + ".out"
        iteration_ends = np.loadtxt(iteration_ends_path)

        k = int(iteration_ends * ratio)

        if critical:
            #find the top k:
            idx = np.argpartition(confs, -k)[-k:]  # Indices not sorted

        #sorted_idxs = idx[np.argsort(confs[idx])][::-1] # Indices sorted by value from largest to smallest
        #print(sorted_idxs)
        #print(confs[sorted_idxs])

        else:
            #find the bottom k:
            idx = np.argpartition(confs, k)[:k]  # Indices not sorted

        #idx[np.argsort(x[idx])]  # Indices sorted by value from smallest to largest

        idx.sort()

        steps_start = idx[0]
        steps_end = idx[0]

        ans = 0
        count = 0

        tmp_end = idx[0]
        tmp_start = idx[0]

        for i in range(1, len(idx)):
        
            # Check if the current element is
            # equal to previous element +1
            if idx[i] == idx[i - 1] + 1:
                count += 1
                tmp_end = idx[i]
                
            # Reset the count
            else:
                count = 0
                tmp_start = idx[i]
                tmp_end = idx[i]
                
            # Update the maximum
            if count > ans:
                ans = count
                steps_start = tmp_start
                steps_end = tmp_end

            

        steps_starts.append(steps_start)
        steps_ends.append(steps_end)

    if critical:
        np.savetxt("./weak_recording/critical_steps_starts.out", steps_starts)
        np.savetxt("./weak_recording/critical_steps_ends.out", steps_ends)
    else:
        np.savetxt("./weak_recording/non_critical_steps_starts.out", steps_starts)
        np.savetxt("./weak_recording/non_critical_steps_ends.out", steps_ends)        


def replay(env, game_num, base_model, step_start, step_end, random_replace, orig_traj_len):
    action_seq = np.load("./weak_recording/act_seq_" + str(game_num) + ".npy", allow_pickle=True)

    if random_replace:
        random_replacement_steps = step_end - step_start
        start_range = int(orig_traj_len - random_replacement_steps)
        step_start = np.random.choice(start_range)
        step_end = step_start + random_replacement_steps
    
    reward = 0
    count = 0

    env.seed(game_num)
    obs = env.reset()

    while True:
        if count < step_start:
            action = action_seq[count]
        elif count <= step_end:
            action = env.action_space.sample()
        else:
            obs = np.clip((obs - vec_normalize.obs_rms.mean) / np.sqrt(vec_normalize.obs_rms.var + vec_normalize.epsilon), - vec_normalize.clip_obs, vec_normalize.clip_obs).astype(np.float32)
            action, _ = base_model.predict(obs, deterministic=True)

        obs, rewards, dones, info = env.step(action)
        reward += rewards[0]

        count += 1
        if dones:
            break
    
    return reward


def cal_fidelity_score(critical_ratios, results, replay_results):
    p_ls = critical_ratios

    p_ds = []

    for j in range(len(p_ls)):
        p_ds.append(np.abs(results[j]-replay_results[j])/4000)

    reward_diff = np.mean(p_ds) if np.mean(p_ds)>0 else 0.001
    fid_score = np.log(np.mean(p_ls)) - np.log(reward_diff)
  
    return fid_score

env = DummyVecEnv([lambda: gym.make("HalfCheetah-v3")])
env = VecMonitor(env, "weak_models/")
base_model = PPO.load("/home/zck7060/Retrain_mujoco/halfcheetah/baseline/weak_models/best_model/best_model")

results = np.loadtxt("./weak_recording/reward_record.out")
ratios = [0.1, 0.2, 0.3, 0.4]
replay_importants = []
fid_scores = []
replay_rand_importants = []
replay_unimportants = []
replay_rand_unimportants = []

for i in range(len(ratios)):
    print("current important threshold: ", ratios[i])

    select_critical_steps(critical=True, ratio=ratios[i])
    
    critical_steps_starts = np.loadtxt("./weak_recording/critical_steps_starts.out")
    critical_steps_ends = np.loadtxt("./weak_recording/critical_steps_ends.out")

    print("Replay(important)")
    critical_ratios = []
    replay_results= []

    for game_num in range(500):
        orig_traj_len = np.loadtxt("./weak_recording/eps_len_"+ str(game_num) + ".out")
        critical_step_start = critical_steps_starts[game_num]
        critical_step_end = critical_steps_ends[game_num]
        critical_ratios.append((critical_step_end - critical_step_start + 1)/orig_traj_len)

        replay_result = replay(env, game_num, base_model, critical_step_start, critical_step_end, \
                            random_replace=False, orig_traj_len=orig_traj_len)
        replay_results.append(replay_result)
    
    print("Average reward: ", np.mean(replay_results))
    replay_importants.append(np.mean(replay_results))

    fid_score = cal_fidelity_score(critical_ratios, results, replay_results)
    fid_scores.append(fid_score)
    print("Fidelity score: ", fid_score)

    print("Replay(rand important)")
    replay_results= []

    for game_num in range(500):
        orig_traj_len = np.loadtxt("./weak_recording/eps_len_"+ str(game_num) + ".out")
        critical_step_start = critical_steps_starts[game_num]
        critical_step_end = critical_steps_ends[game_num]

        replay_result = replay(env, game_num, base_model, critical_step_start, critical_step_end, \
                            random_replace=True, orig_traj_len=orig_traj_len)
        replay_results.append(replay_result)
    
    print("Average reward: ", np.mean(replay_results))
    replay_rand_importants.append(np.mean(replay_results))

    select_critical_steps(critical=False, ratio=ratios[i])
    
    non_critical_steps_starts = np.loadtxt("./weak_recording/non_critical_steps_starts.out")
    non_critical_steps_ends = np.loadtxt("./weak_recording/non_critical_steps_ends.out")

    print("Replay(unimportant)")
    replay_results= []

    for game_num in range(500):
        orig_traj_len = np.loadtxt("./weak_recording/eps_len_"+ str(game_num) + ".out")
        non_critical_step_start = non_critical_steps_starts[game_num]
        non_critical_step_end = non_critical_steps_ends[game_num]

        replay_result = replay(env, game_num, base_model, non_critical_step_start, non_critical_step_end, \
                            random_replace=False, orig_traj_len=orig_traj_len)
        replay_results.append(replay_result)
    
    print("Average reward: ", np.mean(replay_results))
    replay_unimportants.append(np.mean(replay_results))


    print("Replay(rand unimportant)")
    replay_results= []

    for game_num in range(500):
        orig_traj_len = np.loadtxt("./weak_recording/eps_len_"+ str(game_num) + ".out")
        non_critical_step_start = non_critical_steps_starts[game_num]
        non_critical_step_end = non_critical_steps_ends[game_num]

        replay_result = replay(env, game_num, base_model, non_critical_step_start, non_critical_step_end, \
                            random_replace=True, orig_traj_len=orig_traj_len)
        replay_results.append(replay_result)
    
    print("Average reward: ", np.mean(replay_results))
    replay_rand_unimportants.append(np.mean(replay_results))    

print("Ratios: ", ratios)
print("Replay (important): ", replay_importants)
print("Fidelity score: ", fid_scores)
print("Replay (rand important): ", replay_rand_importants)
print("Replay (nonimportant): ", replay_unimportants)
print("Replay (rand nonimportant): ", replay_rand_unimportants)