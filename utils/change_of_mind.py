import numpy as np
import head_angle
import find_frames
import math

from utils.get_values.get_trial_walls import get_trial_walls
from utils.node_coordinates import node_coordinates_at_frame
from utils.manipulate_data.interpolate import interpolate
from utils.video_functions import save_video_trial
from utils.head_angle_to_wall import head_angle_to_wall
from utils.trajectory_speeds import trajectory_speeds, crop_speeds_session
from utils.speed_summary import summary_stats_all_trials
from utils.get_values.get_wall_coords import get_wall_coords
from utils.distance_to_wall import get_wall_node_coords_frame_trial, distance_to_wall

# GLOBALS
# (either side) specificity for attention towards wall
# these params currently seem best for finding CoM trials
ANGLE_SPECIFICITY =  math.pi/12  # math.pi/13   #math.pi/12
# num frames required for attention towards wall
FRAMES_FOR_ATTENTION = 10
# confidence for stopped - multiplier of std below mean for trajectory speeds 
CONFIDENCE = 1.2



def change_of_mind_session(session_trajectories, video_trajectories, labels_file, octpy_metadata, 
                   stim_frames, end_frames, tracks, timestamps_list, video_root, video_filename,
                    color_frame_offset, save_videos=False, inverse=False):
    """ Loop over all trials in session and detect change_of_mind trial
        
        Returns: np.ndarray with trial index values for detected CoM trials
         
          if inverse, return non-CoM indexes """

    # PARAMS
    frames_for_attention = FRAMES_FOR_ATTENTION
    angle_specificity = ANGLE_SPECIFICITY
    filtered_speeds = trajectory_speeds(session_trajectories, timestamps_list)
    concat_speed_mean = summary_stats_all_trials(filtered_speeds)[1]
    concat_speed_std = summary_stats_all_trials(filtered_speeds)[2]
    cropped_speeds = crop_speeds_session(filtered_speeds)
    wall_x_coords, wall_y_coords = get_wall_coords()


    # initialise array for change_of_mind_trial mask
    CoM_trials = np.full([session_trajectories.shape[0]], -1)

    # detect CoM for all trials in session
    for trial in range(session_trajectories.shape[0]):

        CoM_trials[trial] = change_of_mind_trial(trial, session_trajectories, video_trajectories, labels_file, 
                                                 octpy_metadata, stim_frames,
                                                 end_frames, tracks[trial], cropped_speeds[trial], 
                                                 concat_speed_mean, concat_speed_std,
                                                 wall_x_coords, wall_y_coords)

    if not inverse:    
        # return only the indexes where CoM_trials = 1
        CoM_trials_idxs = (CoM_trials==1).nonzero()[0]
    else:
        CoM_trials_idxs = (CoM_trials==0).nonzero()[0]

    if save_videos:
        for trial in CoM_trials_idxs[0:]:
            save_video_trial(video_root, stim_frames, end_frames, trial, labels_file=None,
                             video_filename= video_filename, colorVideo=True, fps=40, 
                                color_frame_offset=color_frame_offset)
            
    return CoM_trials_idxs

def save_CoM_videos(video_root, video_filename, stim_frames, end_frames, CoM_trials_idxs,
                     color_frame_offset=0, labels_file=None, colorVideo=True, fps=40):
    """ Save color camera videos for all trials  in CoM_trials_idxs """

    for trial in range(len(CoM_trials_idxs)):
        save_video_trial(video_root, stim_frames, end_frames, CoM_trials_idxs[trial], labels_file,
                             video_filename=video_filename, colorVideo=colorVideo, fps=fps, 
                                color_frame_offset=color_frame_offset)
        
    
    return None
            
def change_of_mind_trial(trial, session_trajectories, video_trajectories, labels_file, octpy_metadata, 
                   stim_frames, end_frames, track, cropped_speed, concat_speed_mean, concat_speed_std,
                   wall_x_coords, wall_y_coords):
    """ Detect change of mind in a single trial
     
        Returns: 1 if CoM detected, 0 otherwise """

    # PARAMS
    frames_for_attention = FRAMES_FOR_ATTENTION
    angle_specificity = ANGLE_SPECIFICITY
    confidence = CONFIDENCE

    # further left crop speed profile to remove timepoints before mean speed first reached
    # (This is to ensure mouse slows down, but avoid confusion in trials where mouse is initially
    #  stationary)
    if cropped_speed.size > 1:
        mean_trial_speed = np.mean(cropped_speed)
        std_trial_speed = np.std(cropped_speed)
        # find the first index where the mean - 0.75*std in the trial is reached
        for val in cropped_speed:
            if val > (mean_trial_speed - 0.75*std_trial_speed):
                idx = np.where(cropped_speed == val)[0][0]
                break

        cropped_speed = cropped_speed[idx:]

    # # find head_angle, neck_nose_vector, and neck_coords
    # at all frames of the trial
    # find frame nums at stim_on and trial_end
    trial_stim_start_frame = stim_frames[trial]
    trial_end_frame = end_frames[trial]
    trial_frames = np.arange(trial_stim_start_frame, trial_end_frame+1)

    angles_trial = []
    vectors_trial = []
    neck_coords_trial = []
    for frame_num in trial_frames:
        angle, vector = head_angle.extract_head_angle_frame(frame_num, labels_file, track_num=track)
        neck_coords = node_coordinates_at_frame(frame_num, 'neck', video_trajectories, track)
        angles_trial.append(angle)
        vectors_trial.append(vector)
        neck_coords_trial.append(neck_coords)

    # interpolate vectors and angles
    angles_trial = interpolate(angles_trial)
    vectors_trial_array = np.array(vectors_trial)
    vectors_trial_array[:,0] = interpolate(vectors_trial_array[:,0])
    vectors_trial_array[:,1] = interpolate(vectors_trial_array[:,1])


    # find head_angle to high_wall and low_wall at each frame
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
    # have these counters be separate depending on distance to wall
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

    # see if cropped speed profile for the trial every reached below
    # session mean - 1.2*std

    if cropped_speed.size == 1: # occurs if trial could not be cropped
        slowing = False
        # print("No slowing")
    elif cropped_speed[cropped_speed < (concat_speed_mean - concat_speed_std*confidence)].size == 0:
        slowing = False
        # print("No slowing")
    else:
        slowing = True
        # print("Slowing")

    # if at the end of the trial both wall counters are >= 10
    if (wall_1_counter == frames_for_attention) and (wall_2_counter == frames_for_attention):
        # and subject slows during trial to all_trials_speed_mean - 1.2*std
        if slowing == True:
            # this is a change of mind trial
        
            return True
        
    return False
    
def change_of_mind_OLD(session_trajectories, video_trajectories, labels_file, octpy_metadata, 
                   stim_frames, end_frames, tracks, video_root, video_filename,
                    color_frame_offset, save_videos=True):

    # PARAMS
    frames_for_attention = FRAMES_FOR_ATTENTION
    angle_specificity = ANGLE_SPECIFICITY

    # initialise array for change_of_mind_trial mask
    CoM_trial = np.full([session_trajectories.shape[0]], -1)

    # for all trials in a session
    for trial in range(session_trajectories.shape[0]):

        # # find head_angle, neck_nose_vector, and neck_coords
        # at all frames of the trial
        # find frame nums at stim_on and trial_end
        trial_stim_start_frame = stim_frames[trial]
        trial_end_frame = end_frames[trial]
        trial_frames = np.arange(trial_stim_start_frame, trial_end_frame+1)

        angles_trial = []
        vectors_trial = []
        neck_coords_trial = []
        for frame_num in trial_frames:
            angle, vector = head_angle.extract_head_angle_frame(frame_num, labels_file, track_num=tracks[trial])
            neck_coords = node_coordinates_at_frame(frame_num, 'neck', video_trajectories)
            angles_trial.append(angle)
            vectors_trial.append(vector)
            neck_coords_trial.append(neck_coords)

        # interpolate vectors and angles
        angles_trial = interpolate(angles_trial)
        vectors_trial_array = np.array(vectors_trial)
        vectors_trial_array[:,0] = interpolate(vectors_trial_array[:,0])
        vectors_trial_array[:,1] = interpolate(vectors_trial_array[:,1])


        # find head_angle to high_wall and low_wall at each frame
        head_to_wall_1_angs = []
        head_to_wall_2_angs = []
        walls = octpy_metadata.iloc[trial].wall # current trial walls
        trial_type = octpy_metadata.iloc[trial].trial_type # current trial_type
        trial_walls = get_trial_walls(trial_type, walls) # current trial walls as list of ints [high_wall, low_wall]
        for i in range(len(trial_frames)):
            angle_to_wall_1 = head_angle.head_angle_to_wall(vectors_trial[i], trial_walls[0],
                                                    neck_coords_trial[i])
            angle_to_wall_2 = head_angle.head_angle_to_wall(vectors_trial[i], trial_walls[1],
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

    if save_videos:
        for trial in CoM_trial_idxs[0:]:
            save_video_trial(video_root, stim_frames, end_frames, trial, labels_file=None,
                             video_filename= video_filename, colorVideo=True, fps=40, 
                                color_frame_offset=color_frame_offset)