from utils.head_angle_to_wall import head_angle_to_wall
from utils.node_coordinates import node_coordinates_at_frame
from utils.get_values.get_wall_angles import get_wall_angles
import numpy as np
import trajectory_extraction
import find_frames
import head_angle
import sleap
import os
import math

""" Test head angle (to wall) extraction for a single frame in an example trial 
    Seems to be working perfectly"""
# params
wall_num = 3

# paths
# for frame number extraction
data_root = '/home/tomhagley/Documents/SLEAPProject/data'
session = '2022-11-02_A006'

# for trajectory extraction
directory_trajectories = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
file_name_trajectories = 'CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
filepath_trajectories = directory_trajectories + os.sep + file_name_trajectories

# for head angle extraction
labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02_A006_full_predictions_model5_CLI.slp'
labels_file = sleap.load_file(labelsPath)

# extract frames
octpy_metadata, video_metadata, video_metadata_color = find_frames.access_metadata(data_root, session, colorVideo=True)
grey_trials, grey_stims, grey_ends, _, _, _ \
    = find_frames.relevant_session_frames(octpy_metadata, video_metadata, video_metadata_color)

# # extract normalised and flipped/rotated trajectories
# session_trajectories = trajectory_extraction.extract_session_trajectories(filepath_trajectories, \
#                                                                         grey_stims, grey_ends, octpy_metadata, \
#                                                                             normalise=True, flip_rotate=True, smooth=True)




