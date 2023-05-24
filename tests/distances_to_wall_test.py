from utils.distance_to_wall import get_wall_node_coords_trial_start, distance_to_wall_trial_start_sess
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
from utils.correct_color_camera_frame_offset import find_color_camera_frame_offset, find_color_camera_frame_offset_trial
from utils.get_values.get_wall_numbers import get_wall1_wall2
from utils.node_coordinates import node_coordinates_at_frame_session
from utils.head_angle_to_wall import head_angle_to_wall_session
from utils.get_values.get_head_to_wall_angles import get_head_to_wall_angle_session
import numpy as np

""" test distances extraction by plotting trial_start for some trials
    and also the distance to wall1 at this trial start """

# PARAMS
SOCIAL = False
if SOCIAL:
    project_type = 'octagon_multi'
else:
    project_type = 'octagon_solo'

session = '2022-11-04_A004'

# PATHS
# trial metadata
data_root = '/home/tomhagley/Documents/SLEAPProject/data'

# greyscale video
video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}'
video_filename = f'CameraTop_{session}' + '_full.avi'
video_path = video_root + os.sep + video_filename

# color video 
color_video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}'
color_video_filename = f'CameraColorTop_{session}' + '_full.avi'
video_path = color_video_root + os.sep + color_video_filename

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
color_frame_offset = find_color_camera_frame_offset(video_metadata, color_video_metadata, start=False)

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

### TEST DISTANCES ### 
trajectories = trajectory_extraction.extract_session_trajectories(trajectories_filepath, grey_stims,
                                                                  grey_ends, octpy_metadata, tracks, normalise=True,
                                                                  smooth=True)
wall_1_session, wall_2_session = get_wall1_wall2(octpy_metadata, grating_high=True)
node_name = 'BodyUpper'

distances_wall_1 =  distance_to_wall_trial_start_sess(trajectories, node_name, wall_1_session, tracks)
distances_wall_2 =  distance_to_wall_trial_start_sess(trajectories, node_name, wall_2_session, tracks)

print(distances_wall_1[:20])
print(distances_wall_2[:20])

print(wall_1_session[:20])
print(wall_2_session[:20])

for trial in range(2):
    angle, vector = head_angle.extract_head_angle_trial(trial=trial, stim_frames=grey_stims, labels_file=labels_file, track_num=0, plotFlag=True)


# get continuous trajectory for full video (allows indexing by frame_num)
video_trajectories, column_names = trajectory_extraction.extract_video_trajectory(sess.tr, normalise=True, smooth=True)
# get the head angle for a single frame
head_ang, head_vector = head_angle.extract_head_angle_frame(1050, labels_file, track_num=0, plotFlag=True) #1050
# get the (neck) node coordinates for a single frame
neck_coords = node_coordinates_at_frame(1050, 'neck', video_trajectories)
angle_to_wall = head_angle_to_wall(head_vector, wall_num, neck_coords)

### PRINT RESULTS ###
print(f"Head angle is {head_ang:.2f}, or {math.degrees(head_ang):.2f} degrees")
print(f"Wall number is {wall_num}")
wall_angles = get_wall_angles()
print(f"Angle of this wall to horizontal is {wall_angles[wall_num - 1]:.2f}, or {math.degrees(wall_angles[wall_num - 1]):.2f} degrees")
print(f"Neck coords are {neck_coords[0]:.2f}, {neck_coords[1]:.2f}")
print(f"Angle to the wall is {angle_to_wall:.2f}, or {math.degrees(angle_to_wall):.2f} degrees")