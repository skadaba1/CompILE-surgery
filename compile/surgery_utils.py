
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
ROLLOUTS_DIR = "/home/compile/CompILE-surgery/compile/expert-rollouts/drawing_valid.pkl"

def process_out(args_dict):
    model, batch, compile_args = viz_utils.load_model_and_batch(args_dict)
    states, actions, rews, lengths, seeds  = batch
    model.training = False
    outputs = model.forward(states, actions, lengths)
    all_encs, all_recs, all_masks, all_b, all_z = outputs

    # all_encs - 32-dim hidden layer encoding for actions + state
    # all_recs - batch_size x length x state/action encoding 
    # all_b - latent samples and logits for boundary prediction on each step in sequence
    # all_z - latent samples and logits for each segment encoding to latent_dim size
    # z - samples for skills 
    # latent index indicates skill (0 - latent_dim-1, each represents a skill)


    # Grab states and actions
    z, z_idx, boundaries_by_latent, segments_by_latent, latents_by_segment, boundaries_by_episode = viz_utils.get_latent_info(outputs=outputs, lengths=lengths, args=compile_args)
    
    
    x, y, z = xyz_from_tensor(states)
    vel = xyz_from_tensor(actions[:, :, 0:args_dict["state_dim"]])
    acc = xyz_from_tensor(actions[:, :, args_dict["state_dim"]::])
  
    segments = segments_from_latent(boundaries_by_episode)

    # Plot states and actions
    plot_xyz_pos(x, y, z, 0, segments, './fig/state_orig.png')
    plot_vel_and_acc(vel, acc, 0, segments, './fig/actions_orig.png')

    # Get action reconstuctions
    state_reconstr, vel_reconstr, acc_reconstr = get_states_and_actions(all_recs, segments)
    
    # Plot state reconstructions
    x_reconstr, y_reconstr, z_reconstr = state_reconstr
    plot_xyz_pos(x_reconstr, y_reconstr, z_reconstr, 0, segments, './fig/state_reconstr.png')

    # Plot action reconstructions
    plot_vel_and_acc(vel_reconstr, acc_reconstr, 0, segments, './fig/actions_reconstr.png')

    
# Grab states from tensor outputs
def xyz_from_tensor(states):
    #print(states.shape)
    t, x, y, z = np.hsplit(states.squeeze(), states.shape[-1])
    i = x[:, -1]
    return x[:, -1], y[:,-1], z[:,-1]

# Grab segments from latent info
def segments_from_latent(boundaries_by_episode):
    return boundaries_by_episode[0]

# Parse reconstructions
def get_states_and_actions(all_recs, segments):

    action_reconstr_tensor = []
    state_reconstr_tensor = []

    for rec, seg in zip(all_recs, segments):
        start, end = seg
        indices = torch.arange(start, end, 1)

        act = rec[0][:, indices, :]
        state = rec[1][:, indices, :]

        action_reconstr_tensor.append(act)
        state_reconstr_tensor.append(state)

    ## build reconstructions 
    action_reconstr_tensor = torch.hstack(action_reconstr_tensor)
    state_reconstr_tensor = torch.hstack(state_reconstr_tensor)

    x_reconstr, y_reconstr, z_reconstr = xyz_from_tensor(state_reconstr_tensor) 

    vel = action_reconstr_tensor[:, :, 0:args_dict["state_dim"]]
    acc = action_reconstr_tensor[:, :, args_dict["state_dim"]::]

    vel_x_reconstr, vel_y_reconstr, vel_z_reconstr = xyz_from_tensor(vel)
    acc_x_reconstr, acc_y_reconstr, acc_z_reconstr = xyz_from_tensor(acc)

    state_reconstr = (x_reconstr.detach().numpy(), y_reconstr.detach().numpy(), z_reconstr.detach().numpy())
    vel_reconstr = (vel_x_reconstr.detach().numpy(), vel_y_reconstr.detach().numpy(), vel_z_reconstr.detach().numpy())
    acc_reconstr = (acc_x_reconstr.detach().numpy(), acc_y_reconstr.detach().numpy(), acc_z_reconstr.detach().numpy())
    
    return state_reconstr, vel_reconstr, acc_reconstr


# Function to pot xyz positions and optionally states from
def plot_xyz_pos(x, y, z, indx, segments=None, path=None):

  # Create a 3D plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  if(segments):
    # Iterate through segments and plot each with a different color
    for segment in segments:
        start, end = segment
        ax.scatter(x[start:end+1], y[start:end+1], z[start:end+1], label=f'Segment {segment}', marker='o')
  else:
    # Plot the 3D trajectory
    ax.scatter(x, y, z, label=f'Trajectory_samples', marker='o')


  # Label each point with its index
  # for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
  #    ax.text(xi, yi, zi, str(i), color='r')

  # Add labels and title
  ax.set_xlabel('X_pos')
  ax.set_ylabel('Y_pos')
  ax.set_zlabel('Z_pos')
  ax.set_title(f'Trajectory: {indx}')

  # Add legend
  ax.legend(loc='upper left', bbox_to_anchor=(-0.35,0.5))

  # Show the 3D plot
  plt.show()
  plt.savefig(path)

# Function to plot actions on segments
def plot_vel_and_acc(vel, acc, indx, segments=None, path=None):

  vel_x, vel_y, vel_z = [np.ravel(v) for v in vel]
  acc_x, acc_y, acc_z = [np.ravel(a) for a in acc]

  # Create a 3D plot
  fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
  if(segments):
    # Iterate through segments and plot each with a different color
    for segment in segments:
        start, end = segment
        axs[0].scatter(vel_x[start:end+1], vel_y[start:end+1], vel_z[start:end+1], label=f'vel, segment {segment}', marker='o')
        axs[1].scatter(acc_x[start:end+1], acc_y[start:end+1], acc_z[start:end+1], label=f'acc, segment {segment}', marker='o')
  else:
    # Plot the 3D trajectory
    axs[0].scatter(vel_x, vel_y, vel_z, label=f'Trajectory_velociy', marker='o')
    axs[1].scatter(acc_x, acc_y, acc_z, label=f'Trajectory_acceleration', marker='o')

  # Add labels and title for subplot 1
  axs[0].set_xlabel('X_pos')
  axs[0].set_ylabel('Y_pos')
  axs[0].set_zlabel('Z_pos')
  axs[0].set_title(f'Trajectory: {indx}')
  # Add legend
  axs[0].legend(loc='upper left', bbox_to_anchor=(-0.35,0.5))

 # Add labels and title for subplot 2
  axs[1].set_xlabel('X_pos')
  axs[1].set_ylabel('Y_pos')
  axs[1].set_zlabel('Z_pos')
  axs[1].set_title(f'Trajectory: {indx}')
  # Add legend
  axs[1].legend(loc='upper left', bbox_to_anchor=(-0.35,0.5))

  # Show the 3D plot
  plt.show()
  plt.savefig(path)

# plot latents on segments

# plot hidden layer embeddings for states + actions

# plot reconstructioned states on segments

# plot reconstructed actions on segments

# plot skills 

args_dict = {"min_skill_length":MIN_SKILL_LENGTH,
    "env_name":"",
    "rollouts_dir":ROLLOUTS_DIR,
    "num_episodes":2,
    "same_traj":False,
    "sample":True,
    "compile_dir":"results/parking-concrete-4d-state1",
    "state_dim":4,
    "min_skill_length":MIN_SKILL_LENGTH}

process_out(args_dict)