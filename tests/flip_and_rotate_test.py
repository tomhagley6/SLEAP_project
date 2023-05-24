import numpy as np
import math
from utils.get_values.get_trial_walls import get_trial_walls
import find_frames
import trajectory_extraction
import os
from utils.flip_and_rotate_trajectory import *
import h5py

""" test functions with an example vector from a test trial """

# params
data_root = '/home/tomhagley/Documents/SLEAPProject/data'
session = '2022-11-02_A006'
project_type = 'octagon_solo'

trial = 10
vector = np.array([-0.6,0]).reshape((2,1))

# tracking data
trajectories_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/exports/{session}'
trajectories_filename = f'CameraTop_{session}' + '_analysis.h5'
trajectories_filepath = trajectories_root + os.sep + trajectories_filename

# load all data
octpy_metadata, videoMetadata_list, colorVideoMetadata_list \
    = find_frames.access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)

# identify relevant frames from session: all trials
grey_trials, grey_stims, grey_ends, color_trials, color_stims, color_ends \
    = find_frames.relevant_session_frames(octpy_metadata, videoMetadata_list, colorVideoMetadata_list)

# test on single vector
trial_type, walls = octpy_metadata.iloc[trial].trial_type, octpy_metadata.iloc[trial].wall
theta = find_rotation_angle(trial_type, walls)
rot_V = rotation(theta, vector[0], vector[1])
rot_flipped_V = flip_rotated_vector(rot_V, octpy_metadata.iloc[trial].trial_type, high_wall_grating=True)


with h5py.File(trajectories_filepath, 'r') as f:
    dset_names = list(f.keys())
    track_occupancy = f['track_occupancy'][:].T
    locations = f['tracks'][:].T
    node_names = [name.decode().lower() for name in f['node_names'][:]] # decode string from UTF-8 encoding

# test on trajectories of single trial
trajectories, col_names = trajectory_extraction.extract_trajectory(locations, node_names, grey_stims, grey_ends,
                                                                    trial, trial_type, walls, track=0, normalise=True)
rot_angles_array = flip_and_rotate_trajectory(trajectories, trial_type, walls)