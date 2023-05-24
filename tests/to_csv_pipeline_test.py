import os
import sleap
import find_frames
import utils.manipulate_data.data_filtering as data_filtering
import trajectory_extraction
import head_angle
import distances
import head_angle_analysis
from utils.trajectory_speeds import trajectory_speeds
from utils.change_of_mind import change_of_mind_session, save_CoM_videos
from utils.h5_file_extraction import get_node_names, get_locations
from utils.correct_color_camera_frame_offset import find_color_camera_frame_offset
from utils.get_values.get_wall_numbers import get_wall1_wall2
from utils.node_coordinates import node_coordinates_at_frame_session
from utils.head_angle_to_wall import head_angle_to_wall_session
from utils.get_values.get_head_to_wall_angles import get_head_to_wall_angle_session, get_head_to_wall_angle_full_trial_full_session
from utils.distance_to_wall import distance_to_wall_trial_start_sess
from utils.get_values.get_real_RT import response_times_session
from utils.get_values.get_wall_coords import get_wall_coords
from time_to_alignment import time_to_alignment_session
from analysis.logistic_regression import logistic_regression_choose_high
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# PARAMS
SOCIAL = False
if SOCIAL:
    project_type = 'octagon_multi'
else:
    project_type = 'octagon_solo'

session = '2022-11-04_A004'
node_name = 'BodyUpper' # general node to use

# PATHS
# trial metadata
data_root = '/home/tomhagley/Documents/SLEAPProject/data'

# greyscale video
video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/videos/{session}'
video_filename = f'CameraTop_{session}' + '_full.avi'
video_path_greyscale = video_root + os.sep + video_filename

# color video 
color_video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/videos/{session}'
color_video_filename = f'CameraColorTop_{session}' + '_full.avi'
video_path = color_video_root + os.sep + color_video_filename

# tracking data
trajectories_directory = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/exports/{session}'
trajectories_filename = f'CameraTop_{session}' + '_analysis.h5'
trajectories_filepath = trajectories_directory + os.sep + trajectories_filename

# labels file
labels_directory = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/predictions/{session}'
labels_filename = f'CameraTop_{session}' + '_predictions.slp'
labels_filepath = labels_directory + os.sep + labels_filename
labels_file = sleap.load_file(labels_filepath)

# video saving
save_video_output_path = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/analysis'

# load metadata files from HPC using octpy
octpy_metadata, video_metadata, color_video_metadata = find_frames.access_metadata(data_root, session, 
                                                                                   colorVideo=True, refreshFiles=False)

# filter data if required (e.g. RT < 15 s, choice-trials only)
octpy_metadata = data_filtering.choice_trials_only(octpy_metadata)

# find ColorCamera and GreyscaleCamera frame numbers for trial start, stim start, and trial end
grey_trials, grey_stims, grey_ends, \
color_trials, color_stims, color_ends = find_frames.relevant_session_frames(octpy_metadata, 
                                                                                            video_metadata, 
                                                                                            color_video_metadata)

# find color camera frame offset to correct for photodiode bug on octagon_1 (this should always be 0 on octagon 2)
# do this for either start or end (start = False)
color_frame_offset = find_color_camera_frame_offset(video_metadata, color_video_metadata, start=True)

# BODY
### winner mouse track ###
if SOCIAL:
    # TODO find which track contains the winner mouse for each trial and assign to self
    # create a tracks array that records track of winner for each trial
    pass
else:
    tracks = np.zeros(octpy_metadata.shape[0]).astype('int')

### TRAJECTORIES ###
### trial trajectories ###
# find smoothed and normalised (interpolated) trajectories for self for each trial in the session 
# (rotated/flipped if needed)
trajectories = trajectory_extraction.extract_session_trajectories(trajectories_filepath, grey_stims,
                                                                  grey_ends, octpy_metadata, tracks, normalise=True,
                                                                  smooth=True)

### video trajectory ###
# find continuous trajectory across full video
video_trajectories, col_names = trajectory_extraction.extract_video_trajectory(trajectories_filepath, normalise=True,
                                                                    smooth=True)

### HEAD ANGLES ###
# find head angles to horizontal and to relevant walls at the stim_start of trial
### head angle to horizontal ### 
head_angles = head_angle.extract_head_angle_session(grey_stims, labels_file, tracks)

### head angle to high and low wall ### 
head_angles_wall_1, head_angles_wall_2 = get_head_to_wall_angle_session(octpy_metadata, video_trajectories,
                                                                               grey_stims, tracks, labels_file)

### head angle to other mouse ###
if SOCIAL:
    # TODO find head_angle to other mouse for social
    pass


### DISTANCES ### 
# distances to wall
if SOCIAL:
    # TODO find distances to each wall for self and other mouse at start of trial
    pass
else:
    # TODO find distances to each wall for self
    wall_1_session, wall_2_session = get_wall1_wall2(octpy_metadata)
    distances_wall1 =  distance_to_wall_trial_start_sess(trajectories, node_name, wall_1_session, tracks)
    distances_wall2 =  distance_to_wall_trial_start_sess(trajectories, node_name, wall_2_session, tracks)


### TO CSV ###

csv_path = f'/home/tomhagley/Documents/SLEAPProject/extracted_data/{session}'


### TRAJECTORIES ###

### RAW ###

trajectories = trajectory_extraction.extract_session_trajectories(trajectories_filepath, grey_stims,
                                                                  grey_ends, octpy_metadata, tracks, normalise=False,
                                                                  smooth=True)
# trajectories = trajectories.drop('trial_num', axis=1)
# trajectories = trajectories.drop('subj_id', axis=1)
# trajectories = trajectories.drop('date', axis=1)

trajectories.insert(0, 'trial_num', octpy_metadata.trial_num.values)
trajectories.insert(0, 'date', octpy_metadata.date.values)
trajectories.insert(0, 'subj_id', octpy_metadata.subjid.values)

trajectories.to_csv(csv_path + os.sep + 'trajectories_raw.csv')

### NORMALISED ###

trajectories = trajectory_extraction.extract_session_trajectories(trajectories_filepath, grey_stims,
                                                                  grey_ends, octpy_metadata, tracks, normalise=True,
                                                                  smooth=True)


trajectories.insert(0, 'trial_num', octpy_metadata.trial_num.values)
trajectories.insert(0, 'date', octpy_metadata.date.values)
trajectories.insert(0, 'subj_id', octpy_metadata.subjid.values)

trajectories.to_csv(csv_path + os.sep + 'trajectories_normalised.csv')

### NORMALISED_FLIPPED_ROTATED

trajectories = trajectory_extraction.extract_session_trajectories(trajectories_filepath, grey_stims,
                                                                  grey_ends, octpy_metadata, tracks, normalise=True,
                                                                  flip_rotate=True, smooth=True)


trajectories.insert(0, 'trial_num', octpy_metadata.trial_num.values)
trajectories.insert(0, 'date', octpy_metadata.date.values)
trajectories.insert(0, 'subj_id', octpy_metadata.subjid.values)

trajectories.to_csv(csv_path + os.sep + 'trajectories_normalised_rot_flip.csv')

### NON-TRAJECTORY ###
head_angles = np.array(head_angles)
subj_id = octpy_metadata.subjid.values
date = octpy_metadata.date.values
trial_num = octpy_metadata.trial_num.values

wall_x_coords, wall_y_coords = get_wall_coords()

data = {'subj_id':subj_id, 'date':date, 'trial_num':trial_num,
        'head_ang':head_angles, 'head_ang_wall_1':head_angles_wall_1, 'head_ang_wall_2':head_angles_wall_2,
        'distances_wall_1':distances_wall1, 'distances_wall_2':distances_wall2}
df = pd.DataFrame(data)

df.to_csv(csv_path + os.sep + 'sleap_extracted_data.csv')

wall_data = {'wall_x':wall_x_coords, 'wall_y':wall_y_coords}
wall_coords = pd.DataFrame(wall_data)
wall_coords.to_csv(csv_path + os.sep + 'wall_coords.csv')
