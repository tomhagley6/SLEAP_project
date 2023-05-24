import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
import sleap
import cv2 
import re
import utils.manipulate_data.data_filtering as data_filtering
from distances import *
from head_angle import *
import trajectory_extraction
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def logistic_regression_choose_high(chose_high, independent_variables, comparison_column, description_string=''):
    """ feed the binary output term, all independent variables (including bias term)
        and any comparison column to print for comparing against model predictions
        
        Output data must be shape (nTrials,) and independent_variables must be 
        shape (nTrials, nVariables), with each column a variable
        All parameters are np.ndarray
        
        Return the model and print out predictions and comparisons for the final 7 
        rows, as well as model intercept and coefficients"""
    
    # describe model
    if description_string:
        print(description_string,"\n")
    # fit model
    model = LogisticRegression(solver='liblinear', random_state=0).fit(independent_variables, chose_high)
    # print summaries
    print("### Model predictions ###\n", model.predict_proba(np.flip(independent_variables[-7:])))
    print("\n### Model Intercept and Coefficients ###\n", f"Model intercept is: {model.intercept_} \nModel coefficients are: {model.coef_}")
    print("\n### Comparison to selected independent var ###\n", np.flip(comparison_column[-7:],axis=0))

    return model

if __name__ == '__main__':
    """ for one session, find whether oriented closer to high or low wall, find distance
        to walls, in each trial. 
        Fit logistic regression for choosing high to these terms (plus a bias term) """
    
    # PARAMS
    labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02_A006_full_predictions_model5_CLI.slp'
    labels_file = sleap.load_file(labelsPath)
    wall_angles = get_wall_angles()
    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    session = '2022-11-02_A006'

    # find frames in session data
    octpy_metadata, videoMetadata_list, colorVideoMetadata_list = access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)
    octpy_metadata_choice = data_filtering.choice_trials_only(octpy_metadata=octpy_metadata)
    _, stim_frames_choice, _, _, _, _ = relevant_session_frames(octpy_metadata_choice, videoMetadata_list, colorVideoMetadata_list)


    # INDEPENDENT VARIABLES
    # find whether oriented closer to low or high wall
    tracks = np.zeros(stim_frames_choice.shape[0]).astype('int')
    om_choice = head_angle_to_wall_stim_on(octpy_metadata_choice, stim_frames_choice, labels_file, \
                                    tracks, wall_angles)
    # binary classification for closer to low or closer to high
    ang_close_to_low = (om_choice["ang_close_to_low"]
                        .values
                        .reshape(-1,1)
                        )
    
    # using raw head_angle_to_wall as two separate terms
    head_ang_wall_high = (om_choice["head_ang_wall_high"]
                          .values
                          .reshape(-1,1)
                        )   
    head_ang_wall_low = (om_choice["head_ang_wall_low"]
                         .values
                         .reshape(-1,1)
                        )
    
    # normalise head angles to between 0 and 1
    head_ang_wall_high = head_ang_wall_high / math.pi 
    head_ang_wall_low = head_ang_wall_low / math.pi

    # find normalised distance to wall for each trial
    # wall 1 is high and wall 2 is low! for trials that are choiceLG or choiceGL
    distances_wall_1, distances_wall_2 = distance_to_wall_session(octpy_metadata_choice, stim_frames_choice, labels_file)
    distances_wall_1 = np.array(distances_wall_1).reshape(-1,1)
    distances_wall_2 = np.array(distances_wall_2).reshape(-1,1)


    # DEPENDENT VARIABLE
    chose_high = (octpy_metadata_choice['chose_high']
                  .values
                 )
    
    # FIT MODELS AND REPORT

    ### MODEL 1 ###
    # bias + distance_to_high, distance_to_low, head_ang_closer_to_low
    description_string = "bias, distance_to_high, distance_to_low, head_ang_closer_to_low"

    # concatenate independent variables and add a constant bias term
    bias_term = np.ones(distances_wall_1.shape)
    independent_variables = np.hstack([bias_term, distances_wall_1, distances_wall_2, ang_close_to_low])
    
    # fit model 1
    print("\n### MODEL 1 ###")
    model_1 = logistic_regression_choose_high(chose_high, independent_variables, ang_close_to_low, description_string)   

    ### MODEL 2 ###
    # bias + distance_to_high, distance_to_low, head_ang_to_high
    description_string_2 = "bias, distance_to_high, distance_to_low, head_ang_wall_high"

    # concatenate independent variables and add a constant bias term
    bias_term = np.ones(distances_wall_1.shape)
    independent_variables_2 = np.hstack([bias_term, distances_wall_1, distances_wall_2, head_ang_wall_high])
     
    # fit model 2
    print("\n### MODEL 2 ###")
    model_2 = logistic_regression_choose_high(chose_high, independent_variables_2, head_ang_wall_high, description_string_2)   

    ### MODEL 3 ###
    # bias + distance_to_high, distance_to_low, head_ang_to_low
    description_string_3 = "bias, distance_to_high, distance_to_low, head_ang_wall_low"

    # concatenate independent variables and add a constant bias term
    bias_term = np.ones(distances_wall_1.shape)
    independent_variables_3 = np.hstack([bias_term, distances_wall_1, distances_wall_2, head_ang_wall_low])
     
    # fit model 3
    print("\n### MODEL 3 ###")
    model_2 = logistic_regression_choose_high(chose_high, independent_variables_3, head_ang_wall_low, description_string_3)   

    ### MODEL 4 ###
    # bias + distance_to_high + distance_to_low, + head_ang_wall_high + head_ang_wall_low
    description_string_4 = "bias, distance_to_high, distance_to_low, head_ang_wall_high, head_ang_wall_low"

    # concatenate independent variables and add a constant bias term
    bias_term = np.ones(distances_wall_1.shape)
    independent_variables_4 = np.hstack([bias_term, distances_wall_1, distances_wall_2, head_ang_wall_high, head_ang_wall_low])

    # fit model 4
    print("\n### MODEL 4 ###")
    model_4 = logistic_regression_choose_high(chose_high, independent_variables_4, head_ang_wall_high, description_string_4)    



   
