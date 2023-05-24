import numpy as np
import os
import math
import scipy
import find_frames
import trajectory_extraction
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from utils.manipulate_data.smooth_trajectory import smooth_trajectory, smooth_trajectory_from_dataframe
from utils.h5_file_extraction import get_locations, get_node_names
from utils.get_values.get_wall_numbers import get_wall1_wall2

""" Test utils.smooth_trajectory functions """

# params
data_root = '/home/tomhagley/Documents/SLEAPProject/data'
session = '2022-11-02_A006'
trial = 10

# params
sns.set_theme('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]

directory_trajectories = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
file_name_trajectories = 'CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
filepath_trajectories = directory_trajectories + os.sep + file_name_trajectories

# load in session data
octpy_metadata, video_metadata_list, color_video_metadata_list \
    = find_frames.access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)

# extract frames
grey_trials, grey_stims, grey_ends, \
color_trials, color_stims, color_ends  \
    = find_frames.relevant_session_frames(octpy_metadata, video_metadata_list, color_video_metadata_list)

# extract single trial trajectory
trial_type = octpy_metadata.iloc[trial].trial_type
walls = octpy_metadata.iloc[trial].wall
tracks = np.zeros(grey_trials.shape[0]).astype('int')
locations = get_locations(filepath_trajectories)
node_names = get_node_names(filepath_trajectories)
trajectory_single, col_names = trajectory_extraction.extract_trajectory(locations, node_names, grey_stims, grey_ends, trial, \
                                                                    trial_type, walls, tracks[trial], normalise=True, flip_rotate=False)


# extract all trial trajectories (normalised, not rotated)
trajectories_session = trajectory_extraction.extract_session_trajectories( \
                        filepath_trajectories, grey_stims, grey_ends, \
                            octpy_metadata, tracks, normalise=True, flip_rotate=False)

# try smoothing one trajectory (take from dataframe)
trajectory_smoothed = smooth_trajectory_from_dataframe(trajectories_session.iloc[trial])
trajectory = trajectories_session.iloc[trial]
# convert from pd.dataframe to np.ndarray
trajectory = np.stack(trajectory.values).T

# plot
fig, ax = plt.subplots()
ax.plot(trajectory[:,1], label='BodyUpper')
ax.plot(trajectory_smoothed[:,1], label='BodyUpper smoothed')
plt.legend()
plt.show()

# try smoothing one trajectory (taken directly from trajectory_extraction.extract_trajectory)
trajectory_smoothed = smooth_trajectory(trajectory_single)
trajectory = trajectory_single

# plot
fig, ax = plt.subplots()
ax.plot(trajectory[:,1], label='BodyUpper')
ax.plot(trajectory_smoothed[:,1], label='BodyUpper smoothed')
plt.legend()
plt.show()


# also from dataframe smoothed at extraction
wall1_session, wall_2_session = get_wall1_wall2(octpy_metadata)
trajectories_session_smoothed = trajectory_extraction.extract_session_trajectories(filepath_trajectories, \
                                                grey_stims, grey_ends, octpy_metadata, tracks, \
                                                    normalise=True, smooth=True)


trajectory_smoothed = trajectories_session_smoothed.iloc[trial]
trajectory_smoothed = np.stack(trajectory_smoothed.values).T

trajectory = trajectories_session.iloc[trial]
trajectory = np.stack(trajectory.values).T


# plot
fig, ax = plt.subplots()
ax.plot(trajectory[:,1], label='BodyUpper')
ax.plot(trajectory_smoothed[:,1], label='BodyUpper smoothed')
plt.legend()
plt.show()