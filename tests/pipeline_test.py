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
from utils.get_values.get_head_to_wall_angles import get_head_to_wall_angle_trial_starts
from utils.distance_to_wall import distance_to_wall_trial_start_sess
from utils.get_values.get_real_RT import response_times_session
from time_to_alignment import time_to_alignment_session
from logistic_regression import logistic_regression_choose_high
import numpy as np
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
color_video_filename = f'CameraColorTop_{session}' + '_full.avi'
video_path = video_root + os.sep + color_video_filename

# tracking data
trajectories_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/exports/{session}'
trajectories_filename = f'CameraTop_{session}' + '_analysis.h5'
trajectories_filepath = trajectories_root + os.sep + trajectories_filename

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
head_angles_wall_1, head_angles_wall_2 = get_head_to_wall_angle_session(octpy_metadata, trajectories,
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


# find misc interesting trajectory stuff:

#  1. find trajectory speed profiles
timestamps_list = find_frames.timestamps_within_trial(grey_stims, grey_ends, video_metadata)
trial_speed_profiles = trajectory_speeds(trajectories, timestamps_list)

#  2. find trajectory distances
# TODO

#  3. find change-of-mind trials
# use continuous trajectory for the whole video
# now run through change_of_mind function and return CoM trial index values 
# use inverse to switch between CoM and non-CoM
CoM_trial_indexes = change_of_mind_session(trajectories, video_trajectories, labels_file, octpy_metadata,
                                           grey_stims, grey_ends, tracks, timestamps_list, video_root, video_filename,
                                           color_frame_offset=color_frame_offset, save_videos=False, inverse=False)

# create new octpy_metadata with only these trials (or only without)
octpy_metadata_CoM = data_filtering.change_of_mind_trials(trajectories, video_trajectories, labels_file, octpy_metadata,
                                           grey_stims, grey_ends, tracks, timestamps_list, video_root, video_filename,
                                           color_frame_offset=color_frame_offset, save_videos=False)

octpy_metadata_non_CoM = data_filtering.non_change_of_mind_trials(trajectories, video_trajectories, labels_file, octpy_metadata,
                                           grey_stims, grey_ends, tracks, timestamps_list, video_root, video_filename,
                                           color_frame_offset=color_frame_offset, save_videos=False)

# save color video clips for CoM trials
save_CoM_videos(video_root, color_video_filename, color_stims, color_ends, CoM_trial_indexes,
                color_frame_offset)

# save greyscale video clips for CoM trials
# save_CoM_videos(video_root, video_filename, grey_stims, grey_ends, CoM_trial_indexes, colorVideo=False)

#  4. time to align head angle with walls
# first find the head_to_wall_ang for every frame in every trial in session
head_wall_1_ang_all_frames, head_wall_2_ang_all_frames = get_head_to_wall_angle_full_trial_full_session(octpy_metadata, video_trajectories, 
                                                                               grey_stims, grey_ends, tracks, labels_file)
# now find the time to align with either of the two walls (angle specificity is in function file)
times_to_head_wall_alignment = time_to_alignment_session(head_wall_1_ang_all_frames, head_wall_2_ang_all_frames)
# find the trial response times to compare with
response_times_sess = response_times_session(octpy_metadata)



# data analysis stuff:
if SOCIAL:
    #  1. logistic regression models of head angle and distance to high/low of self and other
    pass
else:
    #  1. logistic regression models of head angle and distance to high/low of self
    # define dependent variable:
    
    # identify whether chose high
    # currently assuming grating is high!
    octpy_metadata['chose_high'] = np.where(octpy_metadata['chose_light'] == False, True, False)

    chose_high = (octpy_metadata['chose_high']
                  .values
                 )
    # define independent variables:
    bias_term = np.ones(distances_wall1.shape[0])
    description = "model with distances and head angles to both walls"
    independent_variables = np.hstack([distances_wall1.reshape(-1,1), distances_wall2.reshape(-1,1), 
                                       head_angles_wall_1.reshape(-1,1), head_angles_wall_2.reshape(-1,1)])
    model_1 = logistic_regression_choose_high(chose_high, independent_variables, head_angles_wall_2.reshape(-1,1),
                                              description_string=description)   

    pass


    
