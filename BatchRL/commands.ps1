# (Train and) Evaluate trained agents
# Trained on Euler
python .\BatchRL.py -v -r --train_data train_val --fl 50.0 21.0 25.0 -in 2000000 -bo f t --room_nr 43

# Trained on Remote
python .\BatchRL.py -v -r --train_data all --fl 50.0 --rl_sampling all --hop_eval_data test -in 500000 -bo f t --room_nr 43
python .\BatchRL.py -v -r --train_data all --fl 50.0 --rl_sampling all --hop_eval_data test -in 500000 -bo f t --room_nr 41

# Run Opcua controller
# Test controller
 python BatchRL.py -v -u -bo f t f t f --data_end_date 2020-01-21 --hop_eval_data test --train_data all -fl 50.0 21.0 26.0 --room_nr 41 --rl_sampling all -in 100

# Euler Jobs
# Jobs submitted on Euler (31.01.20)
bsub -n 4 -W 24:00 python BatchRL.py -r -v -bo f f f f --data_end_date 2020-01-21 --hop_eval_data test --train_data all -fl 50.0 21.0 26.0 --room_nr 43 --rl_sampling all
bsub -n 4 -W 24:00 python BatchRL.py -r -v -bo f f f f --data_end_date 2020-01-21 --hop_eval_data test --train_data all -fl 50.0 21.0 26.0 --room_nr 41 --rl_sampling all --sam_heat
bsub -n 4 -W 24:00 python BatchRL.py -r -v -bo f f f f --data_end_date 2020-01-21 --hop_eval_data test --train_data all -fl 50.0 21.0 26.0 --room_nr 43 --rl_sampling all --sam_heat
bsub -n 4 -W 24:00 python BatchRL.py -r -v -bo f f f f --data_end_date 2020-01-21 --hop_eval_data test --train_data all -fl 50.0 21.0 26.0 --room_nr 41 --rl_sampling all

# Other jobs
# Cleanup
python .\BatchRL.py -v -c

