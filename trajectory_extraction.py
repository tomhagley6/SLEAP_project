import find_frames
import head_angle
import utils.manipulate_data.data_filtering as data_filtering
import sleap
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import math
import os
import h5py
from utils.manipulate_data.normalising import normalise_trajectory
from utils.flip_and_rotate_trajectory import flip_and_rotate_trajectory
from utils.manipulate_data.smooth_trajectory import smooth_trajectory, smooth_trajectory_from_dataframe
from utils.get_values.get_trial_walls import get_trial_walls
from utils.manipulate_data.interpolate import interpolate
from utils.h5_file_extraction import get_locations, get_node_names


NUM_NODES = 7

""" Simple functions to extract full trajectory data
    for a trial or for an entire session
    Single trials are returned as a numpy array of shape
    [node_x/y, frames]
    Sessions are returned as a dataframe with node_x/y columns
    and trial rows """

# TODO adapt for social
def extract_session_trajectories(trajectory_file_path, stim_frames, end_frames, octpy_metadata, tracks, normalise=False, flip_rotate=False, smooth=False):
    """ Extract trajectories for each trial in a session
        
        Returns a pandas dataframe of shape (trials, nodes)
        where each row contains arrays of coordinate points
        for each node for the full trial
        (can convert individual rows to numpy arrays using
        np.stack(row.values))  """
    
    # extract locations and node names outside of the trials loop
    with h5py.File(trajectory_file_path, 'r') as f:
        dset_names = list(f.keys())
        track_occupancy = f['track_occupancy'][:].T
        locations = f['tracks'][:].T
        node_names = [name.decode().lower() for name in f['node_names'][:]] # decode string from UTF-8 encoding


    # append each trial's trajectories to a list
    trajectories_list = []
    for trial in range(len(stim_frames)):
        # params
        current_trial = octpy_metadata.iloc[trial]
        walls = current_trial.wall
        trial_type = current_trial.trial_type
        
        trajectories, column_names = extract_trajectory(locations, node_names, stim_frames, \
                                                         end_frames, trial, trial_type, walls, tracks[trial], \
                                                            normalise=normalise, flip_rotate=flip_rotate, smooth=smooth)
        trajectories_list.append(trajectories)

    # list of lists where lists are consecutive node x/ys 
    # nested lists are lists of each trial's data for that node x/y
    nested_trajectories = [[] for i in range(len(column_names))]
    for trial_num in range(len(trajectories_list)):
        current_trial = trajectories_list[trial_num]
        for column_num in range(current_trial.shape[1]):
            current_trial_current_column = current_trial[:, column_num]
            nested_trajectories[column_num].append(current_trial_current_column) 

    # create dataframe for the session where each column is populated by a nested list
    # (one nested list for each column name)
    trajectories_session = pd.DataFrame()
    for i in range(len(column_names)):
        trajectories_session[f"{column_names[i]}"] = nested_trajectories[i]
    
    return trajectories_session

def extract_trajectory(locations, node_names, stim_frames, end_frames, trial, trial_type, walls, track, normalise=False, flip_rotate=False, smooth=False):
    """ Extract trajectories from each node for a single trial
        Returns a numpy array of nodes_x/y * frames 
        
        Requires locations and node_names from utils.h5_file_extraction """

    # extract Sleap HDF5 data by slicing all values in dict entries
    # transpose to covert from column major order
    
    # # Do the below outside of the function to save time
    # with h5py.File(trajectory_file_path, 'r') as f:
    #     dset_names = list(f.keys())
    #     track_occupancy = f['track_occupancy'][:].T
    #     locations = f['tracks'][:].T
    #     node_names = [name.decode() for name in f['node_names'][:]] # decode string from UTF-8 encoding


    # find the start and end frames of the trial
    start_frame = stim_frames.iloc[trial]
    end_frame = end_frames.iloc[trial]

    # preallocate size for trajectories array
    column_names = []
    trajectories = np.zeros((NUM_NODES*2, end_frame - start_frame))

    # for each node, fill the x and y columns in the array
    # interpolate the data at this stage to fill nans (meaning no instances tracked)
    for i in range(locations.shape[1]): 
        column_names.extend([f"{node_names[i]}_x", f"{node_names[i]}_y"])
        trajectories[i*2,:] = interpolate(locations[start_frame:end_frame,i,0,track])
        trajectories[i*2 + 1,:] = interpolate(locations[start_frame:end_frame,i,1,track])
    
    # normalise trajectories from image pixel coordinates to between [-1 1] in x and y
    # using vertical/horizontal walls as limits
    if normalise == True:
        num_frames = end_frame - start_frame
        trajectories = normalise_trajectory(trajectories, NUM_NODES, num_frames)
    
    # rotate (and flip) trajectories to normalise high wall to top (wall 7) 
    # and low wall CW of this 
    if flip_rotate == True:
        trajectories = flip_and_rotate_trajectory(trajectories, trial_type, walls)

    # EDIT
    # makes more sense to return trajectory as shape (n_frames, nodes_x/y), so transpose
    trajectories = trajectories.T

    # smooth trajectories using a 1D Savitzky-Golay filter
    if smooth == True:
        trajectories = smooth_trajectory(trajectories)

    return trajectories, column_names

# TODO This function is slow
def extract_video_trajectory(trajectory_file_path, normalise=False, smooth=False):
    """ Extract continuous trajectory from each node for a full video
        Returns a numpy array of nodes_x/y * frames * tracks"""

    # extract Sleap HDF5 data by slicing all values in dict entries
    # transpose to covert from column major order
    with h5py.File(trajectory_file_path, 'r') as f:
        dset_names = list(f.keys())
        track_occupancy = f['track_occupancy'][:].T
        locations = f['tracks'][:].T
        node_names = [name.decode() for name in f['node_names'][:]] # decode string from UTF-8 encoding

    # find the start and end frames of the trial
    start_frame = 0
    end_frame = track_occupancy.shape[1]

    # find number of tracks (1 track for solo, 2 tracks for social)
    num_tracks = locations.shape[3]

    # preallocate size for trajectories array
    column_names = []
    trajectories_all_instances = np.zeros((end_frame - start_frame, NUM_NODES*2, num_tracks))

    # for each node, fill the x and y columns in the array
    # interpolate the data at this stage to fill nans (no instances tracked)
    for track_num in range(num_tracks):

        # initialise single track trajectories
        trajectories = np.zeros((NUM_NODES*2, end_frame - start_frame))

        for i in range(locations.shape[1]): 
            column_names.extend([f"{node_names[i]}_x", f"{node_names[i]}_y"])
            trajectories[i*2,:] = interpolate(locations[start_frame:end_frame, i, 0, track_num])    # x coord
            trajectories[i*2 + 1,:] = interpolate(locations[start_frame:end_frame, i, 1, track_num])# y coord
        
        # normalise trajectories from image pixel coordinates to between [-1 1] in x and y
        # using vertical/horizontal walls as limits
        if normalise == True:
            num_frames = end_frame - start_frame
            trajectories = normalise_trajectory(trajectories, NUM_NODES, num_frames)
        
        # makes more sense to return trajectory as shape (n_frames, nodes_x/y), so transpose
        trajectories = trajectories.T

        # smooth trajectories using a 1D Savitzky-Golay filter
        if smooth == True:
            trajectories = smooth_trajectory(trajectories)

        # create an array that includes full trajectories for all tracks
        trajectories_all_instances[:,:,track_num] = trajectories
        # trajectories = trajectories[..., np.newaxis]

    
    return trajectories_all_instances, column_names

def tracking_summary(trajectory_file_name, dset_names, locations, track_occupancy, node_names):
    """ Print summaries about the sleap tracking data """

    print("===file_name===")
    print(trajectory_file_name, end='\n\n')

    print("===HDF5 datasets===")
    print(f"{', '.join(dset_names)}", end='\n\n')

    print("===locations data shape===")
    print(locations.shape, end='\n\n')
    frameCount, nodeCount, _, instanceCount = locations.shape

    print("===mistrackitrajectories_sessionng===")
    unoccupiedTracksIdx = np.where(track_occupancy == False)[1]
    print(f"Number of dropout frames: {np.sum(track_occupancy == False)}", end='\n\n')

    print("===nodes===")
    for i, node in enumerate(node_names):
        print(f"{i}: {node}")

    print(track_occupancy.shape)
    missingInstances = locations[unoccupiedTracksIdx]
    trackedLocations = locations[np.where(track_occupancy != False)[1]]
    trackedLocationsIdx = np.where(track_occupancy != False)[0]


if __name__ == '__main__':
    """ find the normalised trajectories for a session and example trial 
        Plot the normalised trajectory for the example trial
        Plot the flipped/roated/normalised trajectory for the example trial
        Plot the smoothed normalised trajectory for example trial"""
    
    # params
    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    session = '2022-11-02_A006'
    trial = 130

    # plotting params
    sns.set_theme('notebook', 'ticks', font_scale=1.2)
    mpl.rcParams['figure.figsize'] = [15,6]

    
    directory_trajectories = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
    file_name_trajectories = 'CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
    filepath_trajectories = directory_trajectories + os.sep + file_name_trajectories

    
    # load in session data
    octpy_metadata, video_metadata_list, color_video_metadata_list \
        = find_frames.access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)

    # filter and extract frames
    octpy_metadata_choice = data_filtering.choice_trials_only(octpy_metadata=octpy_metadata)
    # octpy_metadata_choice = octpy_metadata
    grey_trials, grey_stims, grey_ends, \
    color_trials, color_stims, color_ends  \
        = find_frames.relevant_session_frames(octpy_metadata, video_metadata_list, color_video_metadata_list)

    # extract a single trial trajectory
    trial_type = octpy_metadata.iloc[trial].trial_type
    walls = octpy_metadata.iloc[trial].wall
    locations = get_locations(filepath_trajectories)
    node_names = get_node_names(filepath_trajectories)
    track = 0
    trajectories, col_names = extract_trajectory(locations, node_names, grey_stims, grey_ends, trial, trial_type, walls, track, normalise=True, flip_rotate=True)

    # plot example trial normalised trajectory
    fig, ax = plt.subplots()
    plt.scatter(trajectories[:,0], trajectories[:,1], s=3, c='b')
    plt.axis('scaled')
    ax.set_xlim([-1,1]), ax.set_ylim([-1,1])
    ax.invert_yaxis()
    plt.show()

    # extract all trial trajectories (normalised, not rotated)
    tracks = np.zeros(grey_stims.shape[0]).astype('int')
    trajectories_session = extract_session_trajectories(filepath_trajectories, grey_stims, grey_ends, octpy_metadata, tracks, normalise=True, flip_rotate=False)

    # extract all trial trajectories (normalised and flipped/rotated)
    trajectories_session_flip_rotate = extract_session_trajectories(filepath_trajectories, grey_stims, grey_ends, octpy_metadata, tracks, normalise=True, flip_rotate=True)
    
    # plot example trial normalised trajectory
    fig, ax = plt.subplots()
    fig = plt.figure()
    ax1 = plt.subplot(121)
    plt.scatter(trajectories_session.iloc[trial].Nose_x, trajectories_session.iloc[trial].Nose_y, s=3, c='b')
    plt.axis('scaled')
    ax1.set_xlim([-1,1]), ax1.set_ylim([-1,1])
    ax1.invert_yaxis()
    ax1.set_title("Normalised trajectory")
    
    # plot example trial flipped rotated normalised trajectory
    ax2 = plt.subplot(122)
    plt.scatter(trajectories_session_flip_rotate.iloc[trial].Nose_x, trajectories_session_flip_rotate.iloc[trial].Nose_y, s=3, c='b')
    plt.axis('scaled')
    ax2.set_xlim([-1,1]), ax2.set_ylim([-1,1])
    ax2.invert_yaxis()
    ax2.set_title("Normalised, flipped and rotated trajectory")

    plt.show()

    # # practice smoothing one trajectory
    # trajectory_smoothed = smooth_trajectory_from_dataframe(trajectories_session.iloc[trial])

    # # practice smoothing session trajectories
    # session_trajectories_smoothed = extract_session_trajectories(filepath_trajectories, grey_stims, grey_ends, octpy_metadata, tracks, normalise=True, smooth=True)

    # ### visualise a trial's smoothed trajectory ### 
    # # first directly from non-smoothed dataframe
    # trajectory_smoothed = smooth_trajectory_from_dataframe(trajectories_session.iloc[trial])
    # trajectory = trajectories_session.iloc[trial]
    # # convert from pd.dataframe to np.ndarray
    # trajectory = np.stack(trajectory.values).T

    # # plot
    # fig, ax = plt.subplots()
    # ax.plot(trajectory[:,1], label='BodyUpper')
    # ax.plot(trajectory_smoothed[:,1], label='BodyUpper smoothed')
    # plt.legend()
    # plt.show()

    # # also from dataframe smoothed at extraction
    # trajectory_smoothed = session_trajectories_smoothed.iloc[trial]
    # trajectory_smoothed = np.stack(trajectory_smoothed.values).T
    
    # trajectory = trajectories_session.iloc[trial]
    # trajectory = np.stack(trajectory.values).T


    # # plot
    # fig, ax = plt.subplots()
    # ax.plot(trajectory[:,1], label='BodyUpper')
    # ax.plot(trajectory_smoothed[:,1], label='BodyUpper smoothed')
    # plt.legend()
    # plt.show()

