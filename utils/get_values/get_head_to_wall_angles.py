import head_angle
import numpy as np
from utils.head_angle_to_wall import head_angle_to_wall_session, head_angle_to_wall
from utils.node_coordinates import node_coordinates_at_frame_session, node_coordinates_at_frame, node_coordinates_trial_start_session
from utils.node_coordinates import node_coordinates_trial_start
from utils.get_values.get_wall_numbers import get_wall1_wall2
from utils.get_values.get_trial_walls import get_trial_walls


def get_head_to_wall_angle_session(octpy_metadata, video_trajectories, stim_frames, tracks, labels_file):
    """ given the full video trajectory, trial metadata, and the chosen frames, find the head angle
        to each wall at these frames

        input: 
        octpy_metadata - trial metadata from find_frames
        video_trajectories - full video trajectories from trajectory_extraction
        stim frames - frames to analyse from find_frames (assumed trial structure)
        tracks - track number of winner mouse for each trial
        labels file - sleap data
         
        returns pandas series for head angle to each wall at stim frames, which can be added to metadata dataframe or
         converted to np.ndarray """
    
    # find head vectors for each trial 
    head_vectors = head_angle.extract_head_vector_session(stim_frames, labels_file, tracks)
    # find wall numbers for each trial (trial type extracted from octpy metadata to order the walls)
    wall_1_session, wall_2_session = get_wall1_wall2(octpy_metadata, grating_high=True)
    # find neck node coordinates at each stim start
    node_coordinates = node_coordinates_at_frame_session(stim_frames, 'neck', video_trajectories, tracks)
    # find head angle to high and low wall at each trial stim start
    head_angles_wall_1 = head_angle_to_wall_session(head_vectors, wall_1_session, node_coordinates)
    head_angles_wall_2 = head_angle_to_wall_session(head_vectors, wall_2_session, node_coordinates)

    return np.array(head_angles_wall_1), np.array(head_angles_wall_2)

# UNUSED
# led to incorrect extraction somehow, don't use
def get_head_to_wall_angle_trial_starts_OLD(octpy_metadata, trial_trajectories, stim_frames, tracks, labels_file):
    """ given the trial trajectories and trial metadata, find the head angle
        to each wall at the start of each trial

        input: 
        octpy_metadata - trial metadata from find_frames
        trial_trajectories - trajectories of each trial from trajectory_extraction
        stim frames - frames to analyse from find_frames (assumed trial structure)
        tracks - track number of winner mouse for each trial
        labels file - sleap data
         
        returns pandas series for head angle to each wall at stim frames, which can be added to metadata dataframe or
         converted to np.ndarray """
    
    # find head vectors for each trial 
    head_vectors = head_angle.extract_head_vector_session(stim_frames, labels_file, tracks)
    # find wall numbers for each trial (trial type extracted from octpy metadata to order the walls)
    wall_1_session, wall_2_session = get_wall1_wall2(octpy_metadata, grating_high=True)
    # find neck node coordinates at each stim start
    node_coordinates = node_coordinates_trial_start_session(stim_frames, 'neck', trial_trajectories, tracks)
    # find head angle to high and low wall at each trial stim start
    head_angles_wall_1 = head_angle_to_wall_session(head_vectors, wall_1_session, node_coordinates)
    head_angles_wall_2 = head_angle_to_wall_session(head_vectors, wall_2_session, node_coordinates)

    return np.array(head_angles_wall_1), np.array(head_angles_wall_2)

# to replace old function (directly above) - correct output
def get_head_to_wall_angle_trial_starts_NEW(octpy_metadata, trial_trajectories, stim_frames, tracks, labels_file, grating_high):
    """ given the trial trajectories and trial metadata, find the head angle
        to each wall at the start of each trial

        input: 
        octpy_metadata - trial metadata from find_frames
        trial_trajectories - trajectories of each trial from trajectory_extraction
        stim frames - frames to analyse from find_frames (assumed trial structure)
        tracks - track number of winner mouse for each trial
        labels file - sleap data
         
        returns pandas series for head angle to each wall at stim frames, which can be added to metadata dataframe or
         converted to np.ndarray """
    
    head_to_wall_1_angs = []
    head_to_wall_2_angs = []
    # find wall numbers for each trial (trial type extracted from octpy metadata to order the walls)
    wall_1_session, wall_2_session = get_wall1_wall2(octpy_metadata, grating_high=grating_high)
    for trial in range(stim_frames.shape[0]):
        # find head vectors for each trial 
        _, head_vector = head_angle.extract_head_angle_frame(stim_frames[trial], labels_file, track_num=tracks[trial], plotFlag=False) 
        
        # find neck node coordinates at each stim start
        # avoid using video trajectory to save time
        # neck_coords = node_coordinates_at_frame(stim_frames[trial], 'neck', video_trajectories)
        neck_coords = node_coordinates_trial_start(trial, 'neck', trial_trajectories)
        
        # find head angle to high and low wall at each trial stim start
        head_to_wall_1_angle = head_angle_to_wall(head_vector, wall_1_session[trial], neck_coords)
        head_to_wall_2_angle = head_angle_to_wall(head_vector, wall_2_session[trial], neck_coords)

        # append to a list for the session and return
        head_to_wall_1_angs.append(head_to_wall_1_angle)
        head_to_wall_2_angs.append(head_to_wall_2_angle)

    return np.array(head_to_wall_1_angs), np.array(head_to_wall_2_angs)

def get_head_to_wall_angle_frame(video_trajectories, wall_num, track_num, labels_file, frame_num):
    """ find head angle to wall for a single frame """
    
    # find head vectors for each trial 
    head_vector = head_angle.extract_head_angle_frame(frame_num, labels_file, track_num, plotFlag=False)[1]
    # find neck node coordinates at each stim start
    node_coordinates = node_coordinates_at_frame(frame_num, 'neck', video_trajectories, track=0)
    # find head angle to high and low wall at this frame
    head_to_wall_angle = head_angle_to_wall(head_vector, wall_num, node_coordinates)

    # single value returned here
    return head_to_wall_angle


def get_head_to_wall_angle_full_trial(octpy_metadata, video_trajectories, stim_frames, end_frames, tracks, labels_file, trial, grating_high):
    """ going frame-by-frame, get the head angle to both walls for all frames in the trial """

    head_to_wall_1_angs = []
    head_to_wall_2_angs = []

    start_frame_num = stim_frames.iloc[trial]
    end_frame_num = end_frames.iloc[trial]
    walls = octpy_metadata.iloc[trial].wall
    trial_type = octpy_metadata.iloc[trial].trial_type
    walls = get_trial_walls(trial_type, walls)

    # account for whether light is Low or light is High
    if not grating_high:
        walls = walls[::-1]

    for frame in range(start_frame_num, end_frame_num + 1):
        head_to_wall_1_angs.append(get_head_to_wall_angle_frame(video_trajectories, walls[0], tracks[0], labels_file, frame))
        head_to_wall_2_angs.append(get_head_to_wall_angle_frame(video_trajectories, walls[1], tracks[0], labels_file, frame))

    return np.array(head_to_wall_1_angs), np.array(head_to_wall_2_angs)


def get_head_to_wall_angle_full_trial_full_session(octpy_metadata, video_trajectories, stim_frames, 
                                                   end_frames, tracks, labels_file, grating_high):
    """ wrapper function for full_trial, which returns the head angle to wall for all frames in all trials
        in the session """
    
    head_to_wall_1_angs_trial = []
    head_to_wall_2_angs_trial = []

    for trial in range(stim_frames.shape[0]):
        head_to_wall_1_angs_trial.append(get_head_to_wall_angle_full_trial(octpy_metadata, 
                                                                           video_trajectories, stim_frames, 
                                                                           end_frames, tracks, labels_file, trial, grating_high)[0])
        head_to_wall_2_angs_trial.append(get_head_to_wall_angle_full_trial(octpy_metadata, 
                                                                           video_trajectories, stim_frames, 
                                                                           end_frames, tracks, labels_file, trial, grating_high)[1])
    
    return head_to_wall_1_angs_trial, head_to_wall_2_angs_trial