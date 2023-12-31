
# Generate model outputs and reconstruction 
from ast import arg
from collections import defaultdict, Counter
import torch
from time import time 
import matplotlib.pyplot as plt
import numpy as np
import pickle
from arguments import args
import os
import viz_utils
import skill_utils

MIN_SKILL_LENGTH = 2
ROLLOUTS_DIR = "/home/compile/CompILE-surgery/compile/expert-rollouts/drawing_eval.pkl"

def process_out(args_dict):
    model, batch, compile_args = viz_utils.load_model_and_batch(args_dict)
    states, actions, rews, lengths, seeds  = batch
    model.training = False
    outputs = model.forward(states, actions, lengths)
    z, z_idx, boundaries_by_latent, segments_by_latent, latents_by_segment, boundaries_by_episode = viz_utils.get_latent_info(outputs=outputs, lengths=lengths, args=compile_args)
    print(segments_by_latent)

args_dict = {"min_skill_length":MIN_SKILL_LENGTH,
    "env_name":"",
    "rollouts_dir":ROLLOUTS_DIR,
    "num_episodes":1,
    "same_traj":False,
    "sample":True,
    "compile_dir":"results/parking-concrete-4d-state1",
    "min_skill_length":MIN_SKILL_LENGTH}

process_out(args_dict)