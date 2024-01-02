#!/bin/sh

### Run on Surgery Trajectories ###
sudo python3 train_compile.py  --iterations=250 --rollouts_path_train=expert-rollouts/drawing_train.pkl --rollouts_path_eval=expert-rollouts/drawing_eval.pkl --latent_dist concrete --latent_dim 4 --num_segments 10 --cont_action_dim 8 --prior_rate 50 --mode state+action --run_name parking-concrete-4d-state1 --state_dim 4 --beta_s 1

### Run on Parking Trajectories ###
# sudo python3 train_compile.py  --iterations=2000 --rollouts_path_train=expert-rollouts/parking_train_1683450947.pkl --rollouts_path_eval=expert-rollouts/parking_eval_1683450947.pkl --latent_dist concrete --latent_dim 4 --num_segments 4 --cont_action_dim 2 --prior_rate 10 --mode state+action --run_name parking-concrete-4d-state1 --state_dim 12 --beta_s 1