python -u retrain.py --env "HalfCheetah-v3"\
                     --agent_path "/home/zck7060/new_code/baseline/best_model"\
                     --masknet_path "/home/zck7060/new_code/masknet/best_model"\
                     --vec_norm_path "/home/zck7060/new_code/baseline/vec_normalize.pkl"\
                     --n_envs 20\
                     --seed 0\
                     --go_prob 0.5\
                     --check_freq 200\
                     --bonus 'e3b'\
                     --bonus_scale 0.01