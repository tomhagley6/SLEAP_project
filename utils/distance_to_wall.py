from utils.get_values.get_node_numbers import get_node_numbers
from utils.get_values.get_wall_coords import get_wall_coords
from utils.manipulate_data.normalising import normalise_1D, normalise
import numpy as np

# def distance_to_wall_trial_start(trajectories, node_name, walls):
#     """ get distance to wall at start frames of trial"""

#     # paramas
#     wall_x_coords, wall_y_coords = get_wall_coords()

#     # # find the node numbers for the chosen node
#     # node_num = get_node_numbers(node_name)
#     # node_idx_x = node_num*2
#     # node_idx_y = node_num*2 + 1
    
#     # for each row (trial) in the pandas dataframe 
#     # index the first value of the chosen node (x and y)
#     # also find the wall coordinates
#     node_x, node_y = trajectories[node_name + '_x'], trajectories[node_name + '_y']
#     node_x_trial_frame = []
#     node_y_trial_frame = []
#     wall_x_coords_trial = []
#     wall_y_coords_trial = []
#     for trial in range(node_x.shape[0]):
#         node_x_trial_frame.append(node_x.iloc[trial][0])
#         node_y_trial_frame.append(node_y.iloc[trial][0])

#         wall_x_coords_trial.append()
        
    
def get_wall_node_coords_frame_trial(frame, trajectories, node_name, wall_x_coords, wall_y_coords, wall, track, trial):
    """ get distance to wall at given frame of trial"""
    
    # find the relevant columns of trajectories for this node
    node_x, node_y = trajectories[node_name + '_x'], trajectories[node_name + '_y']
    
    # find node x/y coords at start of trial
    node_x_trial_frame = node_x.iloc[trial][frame]
    node_y_trial_frame = node_y.iloc[trial][frame]

    # find wall x/y coords at start of trial
    wall_x_coord_trial = wall_x_coords[wall - 1]
    wall_y_coord_trial = wall_y_coords[wall - 1]
    
    return node_x_trial_frame, node_y_trial_frame, wall_x_coord_trial, wall_y_coord_trial

def get_wall_node_coords_trial_start(trajectories, node_name, wall_x_coords, wall_y_coords, wall, track, trial):
    """ Get node and wall x/y coordinates at the start frame of trial"""

    
    # find the relevant columns of trajectories for this node
    node_name = node_name.lower()
    node_x, node_y = trajectories[node_name + '_x'], trajectories[node_name + '_y']
    
    # find node x/y coords at start of trial
    node_x_trial_frame = node_x.iloc[trial][0]
    node_y_trial_frame = node_y.iloc[trial][0]

    # find wall x/y coords at start of trial
    wall_x_coord_trial = wall_x_coords[wall - 1]
    wall_y_coord_trial = wall_y_coords[wall - 1]
    
    return node_x_trial_frame, node_y_trial_frame, wall_x_coord_trial, wall_y_coord_trial

def distance_to_wall(node_x_trial_frame, node_y_trial_frame, wall_x_coord_trial, wall_y_coord_trial):
    """ distance to wall for a single point """

    wall_norm = normalise(wall_x_coord_trial, wall_y_coord_trial)
    node_x_norm = node_x_trial_frame
    node_y_norm = node_y_trial_frame

    # find normalised distance between wall and body
    # find x and y differences for pythagoras
    x_diff = abs(wall_norm[0] - node_x_norm)
    y_diff = abs(wall_norm[1] - node_y_norm)

    # pythagoras 
    normalised_distance = np.sqrt(x_diff**2 + y_diff**2)

    # crop distance to be between 0 and 2
    if normalised_distance > 2:
        normalised_distance = 2

    return normalised_distance

def distance_to_wall_trial_start_sess(trajectories, node_name, walls, tracks):
    """ get distance to wall at start frames of trial
    
        Input:
        trajectories - from trajectory_extraction
        node_name - string
        walls - pandas Series of the wall number for each trial in session
        tracks - np.ndarray of track for winner mouse
        (currently trajectories only includes self-mouse, so all tracks are 0)
        
        Returns normalised cartesian distance from node to wall for each trial start frame
        in the session - np.ndarray
        Normalised between 0-2 following convention of coordinate space normalised between
        -1 and 1"""

    wall_x_coords, wall_y_coords = get_wall_coords()


    # find all the x/y coordinates for every trial in session
    # zip plus this list comprehension to assign a full tuple for the whole
    # for loop to the 4 variables
    # asterisk to break up returned list from function into 4 separate parts
    (node_x_trial_frames, 
     node_y_trial_frames, 
     wall_x_coords_trial, 
     wall_y_coords_trial) = zip(*[get_wall_node_coords_trial_start(trajectories, node_name,
                                                             wall_x_coords, wall_y_coords,
                                                             wall=walls[trial], 
                                                             track=tracks[trial],
                                                             trial=trial) 
                                                             for trial in range(trajectories.shape[0])
                                 ]
                                )
    
    # normalise all coordinates
    normalise_1D_vec = np.vectorize(normalise_1D)
    # (already normalised assuming trajectory fed in is already normalised)
    node_x_norm = np.array(node_x_trial_frames)
    node_y_norm = np.array(node_y_trial_frames)
    wall_x_norm = normalise_1D_vec(wall_x_coords_trial, x=True)
    wall_y_norm = normalise_1D_vec(wall_y_coords_trial, x=False)

    # find normalised distance between wall and body
    # find x and y differences for pythagoras
    x_diff = abs(wall_x_norm - node_x_norm)
    y_diff = abs(wall_y_norm - node_y_norm)

    # pythagoras 
    normalised_distance = np.sqrt(x_diff**2 + y_diff**2)

    # crop distance to be between 0 and 2
    normalised_distance[normalised_distance > 2] = 2

    return normalised_distance


def f():
    return 5,6

a,b = zip(*[f() for i in range(10)])



