import numpy as np

if __name__ == '__main__':

    steps_starts = []
    steps_ends = []

    for i_episode in range(10000):
        mask_probs_path = "./weak_retrain_data/mask_probs_" + str(i_episode) + ".out"
        mask_probs = np.loadtxt(mask_probs_path)

        confs = mask_probs

        iteration_ends_path = "./weak_retrain_data/eps_len_" + str(i_episode) + ".out"
        iteration_ends = np.loadtxt(iteration_ends_path)

        k = int(iteration_ends * 0.4)


        #find the top k:
        idx = np.argpartition(confs, -k)[-k:]  # Indices not sorted

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


    np.savetxt("critical_steps_starts.out", steps_starts)
    np.savetxt("critical_steps_ends.out", steps_ends)
      