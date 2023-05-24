import numpy as np
import os
import math
import scipy
import find_frames
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

WINDOW_LENGTH = 25

def smooth_trajectory_from_dataframe(trajectory):
    """ Savitzky-Golay 1D filter a single trial trajectory (all nodes)
        
        Input: single trial (row) from the trajectories pd.dataframe
        
        Output: numpy array of nodes_x/y * frames"""
    
    # params
    window_length = WINDOW_LENGTH

    # trajectories stored as dataframes, so convert to np.ndarray
    trajectory = trajectory.values
    # convert from 1D array of arrays to 2D arr
    # columns are full trajectory for single node _x/_y
    trajectory = np.stack(trajectory).T

    for i in range(trajectory.shape[1]):
        data = trajectory[:,i]
        # can add derivative=1 here to filter the data and also differentiate 
        # This may be the best option because the fitted polynomials are being used
        # to give the differential at each point
        # This might just give back velocity in one shot

        # try window_length WINDOW_LENGTH, unless trial is too short
        try:
            data_filtered = scipy.signal.savgol_filter(data, window_length=window_length, polyorder=3)
            trajectory[:,i] = data_filtered
        except ValueError:
            window_length_corrected = trajectory.shape[0] - 1
            data_filtered = scipy.signal.savgol_filter(data, window_length=window_length_corrected, polyorder=3)
            trajectory[:,i] = data_filtered

    return trajectory

def smooth_trajectory(trajectory):
    """ Savitzky-Golay 1D filter a single trial trajectory (all nodes)
        
        Input: numpy array of n_frames * nodes_x/y
        (output of trajectory_extraction.extract_trajectory)
        
        Output: numpy array of n_frames * nodes_x/y"""
    
    # params
    window_length = WINDOW_LENGTH

    # copy trajectory to avoid changing original array
    trajectory_smoothed = np.copy(trajectory)
    for i in range(trajectory_smoothed.shape[1]):
        data = trajectory_smoothed[:,i]
        # can add derivative=1 here to filter the data and also differentiate 
        # This may be the best option because the fitted polynomials are being used
        # to give the differential at each point
        # This might just give back velocity in one shot

        # try window_length WINDOW_LENGTH, unless trial is too short
        try:
            # use the below for derivative of distance w.r.t time
            # data_filtered = scipy.signal.savgol_filter(data, window_length=window_length, polyorder=3, deriv=1, delta=0.01999998)
            data_filtered = scipy.signal.savgol_filter(data, window_length=window_length, polyorder=3)
            trajectory_smoothed[:,i] = data_filtered
        except ValueError:
            window_length_corrected = trajectory_smoothed.shape[0] - 1

            # account for ValueError: window_length must be odd
            if window_length_corrected % 2 == 0:
                window_length_corrected = window_length_corrected - 1
            
            # account for trajectories of length less than polyorder
            # just return original trajectory
            if window_length_corrected <= 3:
                return trajectory

            data_filtered = scipy.signal.savgol_filter(data, window_length=window_length_corrected, polyorder=3)
            trajectory_smoothed[:,i] = data_filtered

    return trajectory_smoothed

## UNUSED
def smooth_trajectory_session(session_trajectories):
    """ Smooth trajectories for all trials in a session
     
        Input: session_trajectories pd.dataframe output
         from trajectory_extraction.extract_session_trajectories
          
        Output: pd.dataframe object of the same format as input
         (shape = (trials, node_x/y)), with all trajectories
          smoothed by a 1D Savitzky-Golay filter """
    
    # params
    trajectory_list = []
    column_names = session_trajectories.columns
    
    # loop each trial in dataset
    for trial in range(session_trajectories.shape[0]):
        # smooth trial trajectory
        trajectory = smooth_trajectory(session_trajectories.iloc[trial])
        # append each trial's trajectories to a list
        trajectory_list.append(trajectory)

    # Below is for converting from trajectory numpy arrays to session dataframe
    # list of lists where lists are consecutive node x/ys 
    # nested lists are lists of each trial's data for that node x/y
    nested_trajectories = [[] for i in range(len(column_names))]
    for trial_num in range(len(trajectory_list)):
        current_trial = trajectory_list[trial_num]
        for column_num in range(current_trial.shape[0]):
            current_trial_current_column = current_trial[column_num]
            nested_trajectories[column_num].append(current_trial_current_column) 
        

