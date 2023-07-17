import numpy as np

reward_vec = np.loadtxt("./weak_retrain_data/reward_record.out")
lose_reward = np.quantile(reward_vec, .25)
win_reward = np.quantile(reward_vec, .75)

losing_idxs = []
winning_idxs = []
for i in range(10000):
    if reward_vec[i] >= win_reward:
        winning_idxs.append(i)
    elif reward_vec[i] <= lose_reward:
        losing_idxs.append(i)

np.savetxt("losing_game.out", losing_idxs)
np.savetxt("winning_game.out", winning_idxs)