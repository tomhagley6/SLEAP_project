import numpy as np
import os
import math
import find_frames
import trajectory_extraction
import head_angle
import sleap
from utils.get_values.get_trial_walls import get_trial_walls
from utils.manipulate_data.interpolate import interpolate
from utils.node_coordinates import node_coordinates_at_frame
from utils.head_angle_to_wall import head_angle_to_wall
from utils.video_functions import play_video, save_video_trial

""" Identify change of mind trials based on a change in maintained
    head angle from one wall to another """

# GLOBALS
# (either side) specificity for attention towards wall
ANGLE_SPECIFICITY =  math.pi/13  # math.pi/14      #math.pi/12
# num frames required for attention towards wall
FRAMES_FOR_ATTENTION = 10

# PARAMS
trial = 4
track_num = 0
angle_specificity = ANGLE_SPECIFICITY
frames_for_attention = FRAMES_FOR_ATTENTION
color_frame_offset = 52


# PATHS
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

# for video playback
video_root = '/home/tomhagley/Documents/SLEAPProject/octagon_solo'
video_filename = 'CameraColorTop_2022-11-02_A006_full.avi'
output_root = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/analysis'



# BODY
# extract frames
octpy_metadata, video_metadata, video_metadata_color = find_frames.access_metadata(data_root, session, colorVideo=True)
grey_trials, grey_stims, grey_ends, col_trials, col_stims, col_ends \
    = find_frames.relevant_session_frames(octpy_metadata, video_metadata, video_metadata_color)

# create tracks array
tracks = np.zeros(grey_trials.shape[0]).astype('int')

# extract normalised, smoothed, and flipped/rotated trajectories
session_trajectories = trajectory_extraction.extract_session_trajectories(filepath_trajectories,
                                                                        grey_stims, grey_ends, octpy_metadata, tracks,
                                                                            normalise=True, flip_rotate=True, smooth=True)
# extract normalised and smoothed (not flipped/rotated)
video_trajectories, col_names = trajectory_extraction.extract_video_trajectory(filepath_trajectories, normalise=True,
                                                                    smooth=True)

# initialise array for change_of_mind_trial mask
CoM_trial = np.full([session_trajectories.shape[0]], -1)

# for all trials in a session
for trial in range(session_trajectories.shape[0]):

    # # find head_angle, neck_nose_vector, and neck_coords
    # at all frames of the trial
    # find frame nums at stim_on and trial_end
    trial_stim_start_frame = grey_stims[trial]
    trial_end_frame = grey_ends[trial]
    trial_frames = np.arange(trial_stim_start_frame, trial_end_frame+1)

    angles_trial = []
    vectors_trial = []
    neck_coords_trial = []
    for frame_num in trial_frames:
        angle, vector = head_angle.extract_head_angle_frame(frame_num, labels_file, track_num=track_num)
        neck_coords = node_coordinates_at_frame(frame_num, 'neck', video_trajectories)
        angles_trial.append(angle)
        vectors_trial.append(vector)
        neck_coords_trial.append(neck_coords)

    # interpolate vectors and angles
    angles_trial = interpolate(angles_trial)
    vectors_trial_array = np.array(vectors_trial)
    vectors_trial_array[:,0] = interpolate(vectors_trial_array[:,0])
    vectors_trial_array[:,1] = interpolate(vectors_trial_array[:,1])



    # # UNNECESSARY, now done at extraction
    # # interpolate if nans are present
    # if np.NaN in angles_trial:
    #     angles_trial = interpolate(angles_trial)

    # # # REPLACED BY BLOCK BELOW WITH CORRECT ANG_TO_WALL
    # # # find head_angle to high wall and to low wall at all frames
    # wall_angles = head_angle.get_wall_angles() # angles of all walls

    # head_to_wall_1_angs = []
    # head_to_wall_2_angs = []
    # for head_ang in angles_trial:
    #     wall_1_ang, wall_2_ang = head_angle.head_angle_to_wall(head_ang, trial_walls, wall_angles)
    #     head_to_wall_1_angs.append(wall_1_ang)
    #     head_to_wall_2_angs.append(wall_2_ang)

    # to replace the above code:
    head_to_wall_1_angs = []
    head_to_wall_2_angs = []
    walls = octpy_metadata.iloc[trial].wall # current trial walls
    trial_type = octpy_metadata.iloc[trial].trial_type # current trial_type
    trial_walls = get_trial_walls(trial_type, walls) # current trial walls as list of ints [high_wall, low_wall]
    for i in range(len(trial_frames)):
        angle_to_wall_1 = head_angle_to_wall(vectors_trial[i], trial_walls[0],
                                             neck_coords_trial[i])
        angle_to_wall_2 = head_angle_to_wall(vectors_trial[i], trial_walls[1],
                                             neck_coords_trial[i])
        head_to_wall_1_angs.append(angle_to_wall_1)
        head_to_wall_2_angs.append(angle_to_wall_2)


    # # identify change of mind as a head angle towards 
    # # one wall for x frames to the other wall for x frames

    # for angles in trial
    # while angle is within ANGLE_SPECIFICITY degrees of wall
    # increment a counter for that wall
    wall_1_counter = 0
    frame = 0
    while (wall_1_counter < frames_for_attention):
        if frame >= len(head_to_wall_1_angs) - 1:
            break
        if head_to_wall_1_angs[frame] <= angle_specificity:
            wall_1_counter += 1
        else: wall_2_counter = 0
        frame += 1

    wall_2_counter = 0
    frame = 0
    while (wall_2_counter < frames_for_attention):
        if frame >= len(head_to_wall_2_angs) - 1:
            break
        if head_to_wall_2_angs[frame] <= angle_specificity:
            wall_2_counter += 1
        else:
            wall_2_counter = 0

        frame += 1

    # if at the end of the trial both wall counters are >= 10
    # this is a change of mind trial
    if wall_1_counter == frames_for_attention and wall_2_counter == frames_for_attention:
        CoM_trial[trial] = 1
    else:
        CoM_trial[trial] = 0
    # (could add speed heuristic)

CoM_trial_idxs = (CoM_trial==1).nonzero()[0]
print((CoM_trial==1).nonzero())

for trial in CoM_trial_idxs[0:]:
#     save_video_trial(video_root, grey_stims, grey_ends, trial, labels_file,
#                       output_root=output_root, colorVideo=True, fps=False, 
#                       color_frame_offset=color_frame_offset)
    save_video_trial(video_root, col_stims, col_ends, trial, labels_file=None,
                      output_root=output_root, video_filename= video_filename, colorVideo=True, fps=40, 
                      color_frame_offset=color_frame_offset)
play_video('/home/tomhagley/Documents/SLEAPProject/octagon_solo/analysis/part19188-19267.mp4')
    

   # for angles in trial
    # while angle is within ANGLE_SPECIFICITY degrees of wall
    # increment a counter for that wall
    # have these counters be separate depending on distance to wall
    wall_1_counter_near = 0
    wall_1_counter_far = 0
    frame = 0
    while ((wall_1_counter_near < frames_for_attention) and (wall_1_counter_far < int(frames_for_attention*1))):
        
        if frame >= len(head_to_wall_1_angs) - 1:
            break
        
        # find current distance to wall:
        a,b,c,d = get_wall_node_coords_frame_trial(frame, session_trajectories, 'BodyUpper', 
                                         wall_x_coords, wall_y_coords, trial_walls[0], 
                                         track, trial)
        distance_to_wall_1 = distance_to_wall(a,b,c,d)

        # if angle to wall is within acceptance
        if head_to_wall_1_angs[frame] <= angle_specificity:
            
            # increment the counter for near or  far position
            if distance_to_wall_1 > 1:
                wall_1_counter_far += 1
            elif distance_to_wall_1 <= 1:
                wall_1_counter_near +=1
        
        # if angle to wall is not within acceptance, reset both counters
        # as when either reaches threshold, the loop ends
        else: 
            wall_1_counter_near = 0
            wall_1_counter_far = 0

        frame += 1

    wall_2_counter_near = 0 
    wall_2_counter_far = 0
    frame = 0
    while ((wall_2_counter_near < frames_for_attention) and (wall_2_counter_far < int(frames_for_attention*1))):
        
        if frame >= len(head_to_wall_2_angs) - 1:
            break
        
        # find current distance to wall:
        a,b,c,d = get_wall_node_coords_frame_trial(frame, session_trajectories, 'BodyUpper', 
                                         wall_x_coords, wall_y_coords, trial_walls[1], 
                                         track, trial)
        distance_to_wall_2 = distance_to_wall(a,b,c,d)

        # if angle to wall is within acceptance
        if head_to_wall_2_angs[frame] <= angle_specificity:
            
            # increment the counter for near or far position
            if distance_to_wall_2 > 1:
                wall_2_counter_far += 1
            elif distance_to_wall_2 <= 1:
                wall_2_counter_near +=1
        
        # if angle to wall is not within acceptance, reset both counters
        # as when either reaches threshold, the loop ends
        else: 
            wall_2_counter_near = 0
            wall_2_counter_far = 0

        frame += 1

    # see if cropped speed profile for the trial every reached below
    # session mean - 1.2*std

    if cropped_speed.size == 1: # occurs if trial could not be cropped
        slowing = False
        print("No slowing")
    elif cropped_speed[cropped_speed < (concat_speed_mean - concat_speed_std*confidence)].size == 0:
        slowing = False
        print("No slowing")
    else:
        slowing = True
        print("Slowing")

    # if at the end of the trial both wall counters are >= 10
    if ((wall_1_counter_near == frames_for_attention or wall_1_counter_far == int(frames_for_attention*1))
         and (wall_2_counter_near == frames_for_attention or wall_2_counter_far == int(frames_for_attention*1))):
        # and subject slows during trial to all_trials_speed_mean - 1.2*std
        if slowing == True:
            # this is a change of mind trial
        
            return True
        
    return False




