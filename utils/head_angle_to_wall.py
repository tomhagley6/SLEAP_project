import numpy as np
import os
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from utils import angle_between_vectors
from utils.get_values.get_wall_coords import get_wall_coords
from utils.manipulate_data.normalising import normalise
from utils.angle_between_vectors import angle_between_vectors

""" Find the angle between subject head-angle and the wall
    Do this by finding the vector from the head to the wall,
    and then the angle between this and the head-angle vector"""

# TODO check that this always gives smallest angle
def head_angle_to_wall(head_vector, wall_number, subject_ref_coords):
    """ Find the angle between subject head-angle and the wall
        
        Input:
         head_vector: tuple or list normalised vector of neck
                      to nose coordinates
         wall_coords: tuple or list x,y coordinates of the wall
        subject_reference_coords: tuple or list x,y coordinates
                      of reference point for new vector (e.g. neck)
    """
    wall_x_coords, wall_y_coords = get_wall_coords()
    current_wall_coords = []
    current_wall_coords.extend([wall_x_coords[wall_number - 1],
                                wall_y_coords[wall_number - 1]])
    
    # normalise wall coords to compare against reference coords
    current_wall_coords = normalise(current_wall_coords[0],
                                    current_wall_coords[1])


    # define vector for reference point to wall point
    subject_to_wall_vec = (current_wall_coords[0] - subject_ref_coords[0],
                            current_wall_coords[1] - subject_ref_coords[1] )


    # call angle_between vectors on these two
    theta = angle_between_vectors([head_vector[0], head_vector[1], 
                                  subject_to_wall_vec[0],
                                  subject_to_wall_vec[1]])
    
    # return
    return theta

def head_angle_to_wall_session(head_vectors, wall_numbers, subject_ref_coords):
    """ For a list of head vectors, wall_numbers, and reference_coords, find
        the head angle to the relevant wall
        
        (To be used mainly for finding head-angle to wall at a given 
         point in each trial) """
    
    wall_thetas = []
    for trial in range(len(head_vectors)):
        wall_thetas.append(head_angle_to_wall(head_vectors[trial], wall_numbers[trial], subject_ref_coords[trial]))

    return wall_thetas