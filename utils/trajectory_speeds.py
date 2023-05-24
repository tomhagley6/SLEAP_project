import find_frames
import trajectory_extraction
import numpy as np
import scipy
import os
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

""" given a session, extract speeds for all trajectories of all trials 
    Smoothing of trajectories using Savitzky-Golay filter and filtering
    of speed using 1D Gaussian filter"""

# globals
GREYSCALE_FRAMERATE = 50
NODE_X = "BodyUpper_x"
NODE_Y = "BodyUpper_y"


def trajectory_speeds(session_trajectories, timestamps_list):
    """1D Gaussian filtered speeds for all trajectories in a session
     
      Input: 
      session_trajectories - pd.Dataframe from 
      trajectory_extration.extract_session_trajectories, 
      with normalisation and smoothing   
      
      timestamps_list - list of np.ndarrays of timestamps
      for each frame in each trial, output from 
      find_frames.timestamps_within_trial
      
      Output:
      list of np.ndarrays for filtered speeds of BodyUpper point"""
    

    # params
    node_x = NODE_X.lower()
    node_y = NODE_Y.lower()

    # find timedeltas between frames from timestamps list
    session_timedeltas = []
    for trial in range(session_trajectories.shape[0]):
        time_difference = np.diff(timestamps_list[trial])
        time_difference_s = time_difference / np.timedelta64(1, 's') # convert to s
        session_timedeltas.append(time_difference_s)

    # calculate  frame-by-frame speed using timestamps and normalised distance travelled
    # normalised distance travelled:
    session_distances = []
    for trial in range(session_trajectories.shape[0]):
        x_coords = session_trajectories.iloc[trial][f"{node_x}"]
        y_coords = session_trajectories.iloc[trial][f"{node_y}"]

        # find distance travelled - pythagoras
        x_coords_diff = np.diff(x_coords)
        y_coords_diff = np.diff(y_coords)

        x_coords_diff_sq = np.square(x_coords_diff)
        y_coords_diff_sq = np.square(y_coords_diff)

        distances = np.sqrt((x_coords_diff_sq + y_coords_diff_sq).astype(float))
        session_distances.append(distances)

    # speed = distance / time
    session_speeds = []
    for trial in range(session_trajectories.shape[0]):
        session_speeds.append(session_distances[trial] / session_timedeltas[trial])


    # smooth raw frame-by-frame speed with a gaussian filter (sigma = 3, frames = 0.1s used in subgoals paper)
    filtered_speeds = []
    for trial in range(session_trajectories.shape[0]):
        filtered_speed_trial = scipy.ndimage.gaussian_filter(session_speeds[trial], 3)
        filtered_speeds.append(filtered_speed_trial)

    return filtered_speeds

def crop_speed(filtered_speed):
    """ Crop the speed profile of a trial to remove the first and last 0.5 s """

     # params
    greyscale_framerate = GREYSCALE_FRAMERATE

    half_second = int(greyscale_framerate*0.5)

    cropped_speed = filtered_speed[half_second:-half_second]

    # allow smaller crop for < 1s trials
    if cropped_speed.size == 0:
        cropped_speed = filtered_speed[math.ceil(half_second*0.5):-math.ceil(half_second*0.5)]

    # return single value array if still not possible
    if cropped_speed.size == 0:
        return np.array([0])
    
    filtered_speed_length = filtered_speed.size

    # idxs
    start, end = half_second, filtered_speed_length - half_second

    cropped_speed = filtered_speed[start:end]

    return cropped_speed

def crop_speeds_session(filtered_speeds):
    """ for list filtered_speeds, create a new list of 
     cropped speeds with 0.5s removed from start and end """
    
    cropped_speeds_session = []
    for trial in range(len(filtered_speeds)):
        cropped_speeds_session.append(crop_speed(filtered_speeds[trial]))
    
    return cropped_speeds_session
