python -u retrain.py --env "Hopper-v3"\
                     --agent_path "/home/zck7060/Retrain_mujoco/hopper/baseline/weak_tmp/best_model/best_model"\
                     --masknet_path "/home/zck7060/Retrain_mujoco/hopper/masknet/weak_models/best_model/best_model"\
                     --vec_norm_path None\
                     --n_envs 20\
                     --seed 0\
                     --go_prob 0.5\
                     --bonus 'e3b'