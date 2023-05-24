import math
import warnings
import numpy as np
from utils.get_values.get_node_numbers import get_node_numbers


""" Find coordinates of any node at any frame, given session trajectories """

def node_coordinates_at_frame(frame_num, node_name, video_trajectories, track=0):
    """ Find coordinates of any node at any frame, given video trajectories
        Input: 
        frame_num: int, number of frame in camera
        trial: int, number of trial
        node:, int, number of node in skeleton
        session_trajectories: np.ndarray, shape (n_frames, n_nodes_x/y) 
        output of trajectory_extraction.extract_video_trajectories
        track: int, 0 or 1, which subject in trial
         
        Output:
         tuple, (x_coordinate,y_coordinate) of the node """

    # params
    # find node idx from the node_name string
    node_idx = get_node_numbers(node_name)
    node_index_x = node_idx*2
    node_index_y = node_idx*2 + 1
    
    
    # get coords for this node at this frame num in the video
    x_coordinate = video_trajectories[frame_num, node_index_x, track]
    y_coordinate = video_trajectories[frame_num, node_index_y, track]

    # current_trial_node_x = video_trajectories.iloc[trial, node_index_x]
    # current_trial_node_y = video_trajectories.iloc[trial, node_index_y]

    # x_coordinate = current_trial_node_x[frame_num]
    # y_coordinate = current_trial_node_y[frame_num]

    return (x_coordinate, y_coordinate)
    
def node_coordinates_trial_start(trial, node_name, trial_trajectories, track=0):
    """ Find coordinates of any node at trial start frame, given session trajectories
        Input: 
        frame_num: int, number of frame in camera
        trial: int, number of trial
        node:, int, number of node in skeleton
        session_trajectories: pd.Dataframe, shape (trials, n_nodes_x/y) 
        output of trajectory_extraction.extract_session_trajectories
        track: int, 0 or 1, which subject in trial
         
        Output:
         tuple, (x_coordinate,y_coordinate) of the node """

    # params
    # find node idx from the node_name string
    # node_idx = get_node_numbers(node_name)
    # node_index_x = node_idx*2
    # node_index_y = node_idx*2 + 1

    
    # find dataframe column names
    node_x = f"{node_name}_x"
    node_y = f"{node_name}_y"

    # get coords for this node at this frame num in the video
    idx = 0
    x_coordinate = trial_trajectories.iloc[trial][f"{node_x}"][0]
    y_coordinate = trial_trajectories.iloc[trial][f"{node_y}"][0]

    # account for no node found at this frame
    while np.isnan(x_coordinate) or np.isnan(y_coordinate):
        warnings.warn(f"nan returned for coordinate value of node in trial {trial}. Incrementing frame index and retrying.")
        idx += 1
        x_coordinate = trial_trajectories.iloc[trial][f"{node_x}"][idx]
        y_coordinate = trial_trajectories.iloc[trial][f"{node_y}"][idx]

    return (x_coordinate, y_coordinate)

def node_coordinates_at_frame_session(frames_list, node_name, video_trajectories, tracks):
    """ return a list of node_coordinates given a vector of frames
     
        (To be used mainly for finding node coordinates at a given 
         point in each trial) """
    
    node_coordinates = []
    for trial in range(frames_list.shape[0]):
        node_coordinates.append(node_coordinates_at_frame(frames_list[trial], node_name, 
                                                          video_trajectories, track=tracks[trial]))
        
    
    return node_coordinates

def node_coordinates_trial_start_session(start_frames, node_name, trial_trajectories, tracks):
    """ return a list of node_coordinates at initial trial frame
     
        (To be used mainly for finding node coordinates at a given 
         point in each trial) """
    
    node_coordinates = []
    for trial in range(start_frames.shape[0]):
        node_coordinates.append(node_coordinates_trial_start(trial, node_name, 
                                                          trial_trajectories, track=tracks[trial]))
        
    
    return node_coordinates