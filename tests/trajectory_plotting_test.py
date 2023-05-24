import sleap
from trajectory_extraction import extract_session_trajectories, extract_trajectory
import utils.manipulate_data.data_filtering as data_filtering
import find_frames
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import numpy as np
from utils.unused.ordinal_index import ordinal_index
from head_angle import extract_head_angle_trial
from utils.h5_file_extraction import get_locations, get_node_names
from utils.plotting.plot_all_trajectories import plot_all_trajectories
from utils.plotting.plot_start_points import plot_start_points
from utils.plotting.plot_end_points import plot_end_points
from utils.plotting.plot_trajectories_separate_low import plot_all_trajectories_separate_low
import trajectory_extraction

# TODO plotting function to plot specific subset of trajectories
    
# PARAMS

# plotting params
sns.set_theme('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]

SOCIAL = False
if SOCIAL:
    project_type = 'octagon_multi'
else:
    project_type = 'octagon_solo'

session = '2022-11-04_A004'
node_name = 'BodyUpper' # general node to use
trial = 10

# PATHS

# trial metadata
data_root = '/home/tomhagley/Documents/SLEAPProject/data'

# greyscale video
video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/videos/{session}'
video_filename = f'CameraTop_{session}' + '_all.avi'
video_path_greyscale = video_root + os.sep + video_filename

# tracking data
trajectories_directory = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/exports/{session}'
trajectories_filename = f'CameraTop_{session}' + '_analysis.h5'
trajectories_filepath = trajectories_directory + os.sep + trajectories_filename

# labels file
labels_directory = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/predictions/{session}'
labels_filename = f'CameraTop_{session}' + '_predictions.slp'
labels_filepath = labels_directory + os.sep + labels_filename
labels_file = sleap.load_file(labels_filepath)


# LOADING DATA

# load metadata files from HPC using octpy
octpy_metadata, video_metadata, color_video_metadata = find_frames.access_metadata(data_root, session, 
                                                                                   colorVideo=True, refreshFiles=False)

# filter data if required (e.g. RT < 15 s, choice-trials only)
octpy_metadata = data_filtering.choice_trials_only(octpy_metadata)
octpy_metadata = data_filtering.sub_x_RT_only(octpy_metadata, 15)


# find ColorCamera and GreyscaleCamera frame numbers for trial start, stim start, and trial end
grey_trials, grey_stims, grey_ends, \
color_trials, color_stims, color_ends = find_frames.relevant_session_frames(octpy_metadata, 
                                                                                            video_metadata, 
                                                                                            color_video_metadata)

tracks = np.zeros(octpy_metadata.shape[0]).astype('int')

### TRAJECTORIES ###
### trial trajectories ###
# find smoothed and normalised (interpolated) trajectories for self for each trial in the session 
# (rotated/flipped if needed)
trajectories = trajectory_extraction.extract_session_trajectories(trajectories_filepath, grey_stims,
                                                                  grey_ends, octpy_metadata, tracks, normalise=True,
                                                                  smooth=True, flip_rotate=True)
trial_type = octpy_metadata.iloc[trial].trial_type
walls = octpy_metadata.iloc[trial].wall
locations = get_locations(trajectories_filepath)
node_names = get_node_names(trajectories_filepath)
# trajectories, col_names = extract_trajectory(locations, node_names, grey_stims, grey_ends, trial, trial_type, walls, 0)

# BODY

# # plot all trajectories
plot_all_trajectories(video_path_greyscale, trajectories, node_name, title='All trajectories')

# # plot trajectory start points
plot_start_points(trajectories, node_name, title='Trial start locations')

# # plot all end points
plot_end_points(trajectories, node_name, title='Trial end locations')

# plot all trajectories (separate choose_high from choose_low)
plot_all_trajectories_separate_low(octpy_metadata, video_path_greyscale, trajectories, node_name, title='All trajectories')
