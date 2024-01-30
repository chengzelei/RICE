python -u retrain.py --env "Hopper-v3"\
                     --agent_path "baseline/best_model"\
                     --masknet_path "masknet/best_model"\
                     --vec_norm_path "baseline/vec_normalize.pkl"\
                     --n_envs 20\
                     --seed 0\
                     --go_prob 1\
                     --check_freq 200\
                     --bonus 'rnd'\
                     --bonus_scale 0
