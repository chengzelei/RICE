import numpy as np
import os

def calculate_fidelity_score(path,method,width):
    reward_log = np.loadtxt(path + "reward_log.txt")
    replay_reward_log = np.loadtxt(path + str(method) + "_replay_reward_log_" + str(int(width * 100)) + ".txt")
    score = 0
    for i in range(replay_reward_log.shape[0]):
        score += np.abs(reward_log[i] - replay_reward_log[i]) / 100 - np.log(width)
    average_score = score / replay_reward_log.shape[0]
    print("Method: {}, Width: {}, Fidelity Score: {}".format(method, width, average_score))
    


if __name__ == '__main__':
    methods = ["random", "trust", "highlights", "mask","max"]
    # methods = ["random", "trust", "highlights"]
    widths = [0.02,0.04,0.06,0.08]
    # widths = [0.02,0.03,0.04,0.05,0.10,0.15,0.20]
    path = "recordings/"
    for width in widths:
        for method in methods:
            calculate_fidelity_score(path,method,width)
        print("===========================================")