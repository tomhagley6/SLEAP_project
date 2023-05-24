import numpy as np
import warnings

def normalise(x,y):
    """ normalise any x-y coordinate point to be between [-1, 1]
     where bounds are the most extreme coordinates within the 
      octagon walls """
    # normalise distance using centre and top-left coordinates
    centre = np.array((699, 532))
    top_left = np.array((227, 66))
    point = np.array([x, y])
    half_arena = centre - top_left

    diff = point - top_left
    normalised_diff = diff/(2*half_arena)

    for i in range(2):
        if normalised_diff[i] > 1:
            normalised_diff[i] == 1
    
    # convert to [-1,1] range
    normalised_diff = normalised_diff*2 - 1

    return normalised_diff

def normalise_1D(coord, x=True):
    """ Normalise a single x or y coordinate
        This function allows vectorising to convert full trajectories
         
        first use np.vectorize, then feed in a 1D vector of either
         x or y points (specify with x=True or x=False) """
    # normalise distance using centre and top-left coordinates
    centre = np.array((699, 532))
    top_left = np.array((227, 66))

    if x:
        centre = centre[0]
        top_left = top_left[0]
        point = coord
    else:
        centre = centre[1]
        top_left = top_left[1]
        point = coord

    half_arena = centre - top_left
    diff = point - top_left
    normalised_diff = diff/(2*half_arena)


    if normalised_diff > 1:
        normalised_diff = 1
    elif normalised_diff < 0:
        normalised_diff = 0

    # convert to [-1,1] range
    normalised_diff = normalised_diff*2 - 1
    
    return float(normalised_diff)


def normalise_trajectory(trajectory, num_nodes, num_frames):
    """ normalise a full trajectory """

    # vectorise normalisation functions
    normalise_1D_vec = np.vectorize(normalise_1D)

    # for each node_x/y in the trajectory array, normalise each point
    normalised_trajectory = np.zeros((num_nodes*2, num_frames))
    for i in range(num_nodes*2):
        if i % 2 == 0: # if even, x coordinate (i = 0 + 2k is an x)
            normalised_x_vector = normalise_1D_vec(trajectory[i], x=True)
            normalised_trajectory[i,:] = normalised_x_vector
        elif i % 2 == 1: # if odd, y coordinate (i = 1 + 2k is a y)
            normalised_y_vector = normalise_1D_vec(trajectory[i], x=False)
            normalised_trajectory[i,:] = normalised_y_vector
        else:
            warnings.warn("Index error for normalising trajectory")

    return normalised_trajectory
        

def normalise_session_trajectories(session, num_nodes):
    """ normalise trajectories of full session data """

    session_trajectories_list = []
    for i in range(session.shape[0]):
        num_frames = session.iloc[i].shape
        trajectory = normalise_trajectory(session.iloc[i], num_nodes, num_frames)
        session_trajectories_list.append(trajectory)

    # now create a pandas dataframe