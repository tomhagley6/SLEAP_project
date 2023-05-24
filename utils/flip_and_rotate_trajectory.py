import numpy as np
import math
from utils.get_values.get_trial_walls import get_trial_walls
import find_frames
import os

""" Functions to rotate and flip position vectors so that 
    high wall is at WALL_HIGH_NUM and low wall is at WALL_LOW_NUm.
    Input for non-vectorised functions is an x-y coordinate vector """

WALL_HIGH_NUM = 7
WALL_LOW_NUM = 8
NUM_WALLS = 8


# TODO Check functions work on example vector
# TODO Either:
#       make umbrella function and vectorise this for trajectory
#       or make trajectory function and vectorise all individual functions within it

def flip_and_rotate_trajectory(trajectories, trial_type, walls):
    """ flip and rotate a single trajectory (given by trajectories),
        using the trial_type and the walls
         
        Return the flipped and rotated trajectory """
    # params
    num_nodes_x_y, num_frames = trajectories.shape[0], trajectories.shape[1]
    num_nodes = int(num_nodes_x_y/2) # skeleton nodes have x and y coords stored separately
    #trial_type = octpy_metadata.iloc[trial].trial_type

    # vectorise functions
    # set the output datatype or you'll get weird errors... 
    rotation_vectorised = np.vectorize(rotation, otypes=[np.ndarray])
    flip_rotated_vector_vectorised = np.vectorize(flip_rotated_vector, otypes=[np.ndarray])

    # set this array to object type on initialisation to avoid errors
    # when trying to assign np.ndarray (object) types to it
    flipped_rotated_angles_array = np.zeros((num_nodes*2, num_frames)).astype('O')
    
    # repeat for each node in skeleton
    for i in range(num_nodes):
        # get x and y coordinate vectors
        x_v = trajectories[i*2, :]
        y_v = trajectories[i*2+1, :]
        # rotation angle for this trial
        rotation_angle_trial = find_rotation_angle(trial_type, walls)
        # apply rotation to array of vectors
        rotated_vectors_trial = rotation_vectorised(rotation_angle_trial, x_v, y_v)
        # flip 
        flipped_rotated_vectors_trial = flip_rotated_vector_vectorised(rotated_vectors_trial, trial_type)

        # separate x and y coordinates to fit format of trajectories dataframe
        # convert from 1D array of arrays to multidimensional array
        flipped_rotated_vectors_trial = np.stack(flipped_rotated_vectors_trial)
        # reshape to 2D from 3D (as vectors were stored vertically)
        flipped_rotated_vectors_trial = flipped_rotated_vectors_trial.reshape((flipped_rotated_vectors_trial.shape[0], -1))

        # can now split the two columns
        flipped_rotated_angles_array[i*2,:] = flipped_rotated_vectors_trial[:,0]
        flipped_rotated_angles_array[i*2 + 1,:] = flipped_rotated_vectors_trial[:,1]

    return flipped_rotated_angles_array
    
# currently just returning CW angle, because flipping the y-axis to
# plot top left as (-1, -1)
def find_rotation_angle(trial_type, walls):
    """ Find CCW angle of rotation for vector to 
        rotate arena s.t. CCW wall is at position
        WALL_HIGH_NUM and CW wall is at position
        WALL_LOW_NUM """
    
    # params
    wall_high = WALL_HIGH_NUM
    num_walls = NUM_WALLS
    # take flipped walls because we just want high wall at
    # WALL_HIGH_NUM - it doesn't matter which side low is 
    walls = get_trial_walls(trial_type, walls)
    unitary_rot_ang = 2*math.pi / num_walls

    # find difference between WALL_HIGH_NUM and high wall 
    if walls[0] <= wall_high:
        difference = wall_high - walls[0]
    else: # account for negative difference
        difference = wall_high - (walls[0] - num_walls)

    # CW angle
    rot_ang = unitary_rot_ang * difference 

    # # Currently removed to fit image plotting coordinates
    # # CCW angle
    # rot_ang = 2*math.pi - rot_ang

    return rot_ang


def rotation(theta, x, y):
    """ Take counterclockwise rotation angle and starting vector
        Return rotated vector """
    
    vector = np.array([x,y]).reshape(2,1)

    rotM = np.array([
                    [math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]
                    ])
    return np.matmul(rotM, vector)


def flip_rotated_vector(vector, trial_type, high_wall_grating=True):
    """ flip rotated vector if high wall is CW instead of
         CCW.
        Assumes vector already rotated """
    
    # trial_type = octpy_metadata.iloc[trial].trial_type
    
    if high_wall_grating:
        if trial_type == 'choiceLG': # flip vector so low is CW
            vector = np.array([-vector[0], vector[1]]).reshape((2,1))
        elif trial_type == 'choiceGL': # vector doesn't need flipping
            pass
    else:
        if trial_type == 'choiceLG': # vector doesn't need flipping
            pass
        elif trial_type == 'choiceGL': # flip vector so low is CW
            vector = np.array([-vector[0], vector[1]]).reshape((2,1))

    return vector


## TESTING ONLY
if __name__ == '__main__':
    """ test functions with an example vector from a test trial """

    # params
    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    data_type = 'aeon_mecha'
    session = '2022-11-02_A006'
    trial = 10
    vector = np.array([-0.6,0]).reshape((2,1))

    directory_trajectories = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
    file_name_trajectories = 'CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
    filepath_trajectories = directory_trajectories + os.sep + file_name_trajectories

    # load all data
    octpy_metadata, videoMetadata_list, colorVideoMetadata_list \
        = find_frames.access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)

    # identify relevant frames from session: all trials
    grey_trials, grey_stims, grey_ends, color_trials, color_stims, color_ends \
        = find_frames.relevant_session_frames(octpy_metadata, videoMetadata_list, colorVideoMetadata_list)
    
    # test on single vector
    trial_type, walls = octpy_metadata.iloc[trial].trial_type, octpy_metadata.iloc[trial].wall
    theta = find_rotation_angle(trial_type, walls)
    rot_V = rotation(theta, vector[0], vector[1])
    rot_flipped_V = flip_rotated_vector(rot_V, octpy_metadata.iloc[trial].trial_type, high_wall_grating=True)

    # # test on trajectories of single trial
    # trajectories, col_names = trajectory_extraction.extract_trajectory(filepath_trajectories, grey_stims, grey_ends, trial, normalise=True)
    # rot_angles_array = flip_and_rotate_trajectory(octpy_metadata, 10, trajectories)