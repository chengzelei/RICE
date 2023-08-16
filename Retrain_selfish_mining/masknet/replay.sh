nohup python -u replay.py --method "random" > run_logs/1.log 2>&1 &
nohup python -u replay.py --method "mask" > run_logs/2.log 2>&1 &
nohup python -u replay.py --method "trust" > run_logs/3.log 2>&1 &
nohup python -u replay.py --method "highlights" > run_logs/4.log 2>&1 &
nohup python -u replay.py --method "max" > run_logs/5.log 2>&1 &