#!/bin/bash  

python -u retrain.py --n_envs 10\
                    --seed 2\
                    --bonus_scale 1e-2\
                    --go_prob 0.8
