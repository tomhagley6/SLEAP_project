import os
import sleap
import find_frames
import utils.manipulate_data.data_filtering as data_filtering
import trajectory_extraction
import head_angle
import distances
import head_angle_analysis
from utils.trajectory_speeds import trajectory_speeds
from utils.change_of_mind import change_of_mind_session, save_CoM_videos
from utils.h5_file_extraction import get_node_names, get_locations
from utils.correct_color_camera_frame_offset import find_color_camera_frame_offset
from utils.get_values.get_wall_numbers import get_wall1_wall2
from utils.node_coordinates import node_coordinates_at_frame_session
from utils.head_angle_to_wall import head_angle_to_wall_session
from utils.get_values.get_head_to_wall_angles import get_head_to_wall_angle_session, get_head_to_wall_angle_full_trial_full_session
from utils.distance_to_wall import distance_to_wall_trial_start_sess
from utils.get_values.get_real_RT import response_times_session
from time_to_alignment import time_to_alignment_session
from analysis.logistic_regression import logistic_regression_choose_high
from utils.plotting.heatmap import heatmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from session import Session
import math
import seaborn as sns

""" Script to generate logistic regression models of all independent variable combinations,
    trained on experimental data. Then, predict on simulated data and visualise as heatmaps
    This script is for distances """

# GLOBALS
ITERATIONS = 10000          # simulated data iters
SOCIAL = False              # social session
GRATING_HIGH = True         # is grating High reward
PRINT_ALL = False           # print summaries of all models
BIN_MAX = 180             # x,y max bin values for the heatmap


# FUNCTIONS
def train(data, xvars, yvar, drop_vars):
    """ Train full and dropped-term logistic regression models
        Assuming bias term and 4 other terms (although easily changeable)
        Create all possible combinations of logistic regression models 
        Use statsmodels module for logistic regression functions

        Use 'predict_and_visualise' function for predicting on simulated data and
        heatmap visualisations
      
       Input:
        data - dictionary of column_name:column_data as type string:np.ndarray
        xvars - list of strings of independent variable column_name
        yvar - string of dependent variable column_name
        drop_vars - list of strings of columns_name for independent variables to drop

        
        Output:
         log_reg - full model
         dropped_models - all modules with 1+ terms dropped (SUM(between k=1 and k=2) nCk)
         dropped_models_info - description of the model, in the same order as dropped models
                                (use this to navigate)
         dropped_models_LRT - likelihood ratio test p-value for all dropped models compared to 
                                full model
        """
    
    # define Xtrain (IVs) and ytrain (DV)
    data = pd.DataFrame(data)

    Xtrain = data[xvars]
    ytrain = data[yvar]

    # print summary
    print('\n\n### STATSMODELS LOGISTIC REGRESSION ###')
    log_reg = sm.Logit(ytrain, Xtrain).fit()
    print(log_reg.summary())
    print(log_reg.summary2())

    # drop single terms from the model 
    IV = data.drop('chose_high', axis=1)
    #IV = IV.drop('bias_term', axis=1) ## uncomment to remove bias term


    # lists to save: the model, description of model, likelihood ratio test outcome of model
    # and unique barcode for terms used in model
    dropped_models = []
    dropped_models_info = []
    dropped_models_LRT = []
    dropped_vars_dicts_list = []
    idx_count = -1
    
    # nested for loops to run through all combinations of model terms in producing possible models
    # only a single model is saved for each unique combination of model terms
    # if the number of terms increases, can increase for loops here to be n-1 loops
    for var1 in drop_vars:
        for var2 in drop_vars:
            for var3 in drop_vars:
                # Xtrain = [x for x in data.columns if x is not f"{var}" and x is not 'chose_high' and x is not 'bias_term']
                Xtrain = [x for x in data.columns if x is not f"{var1}" and x is not f"{var2}" and x is not f"{var3}" and x is not 'chose_high']
                Xtrain = data[Xtrain]
                ytrain = data['chose_high']
                model_drop = sm.Logit(ytrain, Xtrain).fit()
                
                # dropped_vars = set([var1, var2, var3])
                dropped_vars = {var:None for var in [var1, var2, var3]} # use dict instead of set to maintain order
                idx_count+=1 # save index value with description

                # if all dropped variables are the same (only dropping one term)
                if len(dropped_vars) == 1:
                    dropped_vars_list = list(dropped_vars) # convert back to list to index
                    print(f"Drop {dropped_vars_list[0]}:")
                    # find log likelihood comparison for likelihood ratio test
                    # comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
                    LRT = 2 * ((-model_drop.llf) - (-log_reg.llf)) 
                    # LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
                    # survival function is (1 - cdf) for LRT value given degrees of freedom  
                    p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop(dropped_vars_list[0], axis=1).shape[1])

                    # save model and info to list 
                    description_string = f"{idx_count} - Dropped {dropped_vars_list[0]} from full model"
                    if dropped_vars in dropped_vars_dicts_list:
                        idx_count-= 1
                        continue
                    else:
                        dropped_models.append(model_drop)
                        dropped_models_info.append(description_string)
                        dropped_models_LRT.append(p_value)
                        dropped_vars_dicts_list.append(dropped_vars)

                    # print(model_drop.summary())
                    # print(f"LRT statistic: {LRT:.4f}")
                    # print(f"P-value: {p_value:.4f}\n")

                # if two dropped variables are the same (dropping 2 terms)
                elif len(dropped_vars) == 2:
                    dropped_vars_list = list(dropped_vars) # convert back to list to index
                    print(f"Drop {dropped_vars_list[0]} and {dropped_vars_list[1]}:")
                    # find log likelihood comparison for likelihood ratio test
                    # comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
                    LRT = 2 * ((-model_drop.llf) - (-log_reg.llf)) 
                    # LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
                    # survival function is (1 - cdf) for LRT value given degrees of freedom  
                    p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop([dropped_vars_list[0], dropped_vars_list[1]], axis=1).shape[1])

                    # save model and info to list 
                    description_string = f"{idx_count} - Dropped {dropped_vars_list[0]} and {dropped_vars_list[1]} from full model"
                    if dropped_vars in dropped_vars_dicts_list:
                        idx_count -= 1
                        continue
                    else:
                        dropped_models.append(model_drop)
                        dropped_models_info.append(description_string)
                        dropped_models_LRT.append(p_value)
                        dropped_vars_dicts_list.append(dropped_vars)

                    # print(model_drop.summary())
                    # print(f"LRT statistic: {LRT:.4f}")
                    # print(f"P-value: {p_value:.4f}\n")

                # all dropped variables are unique, dropping 3 terms from the full model (one term remaining)
                elif len(dropped_vars) == 3:
                    dropped_vars_list = list(dropped_vars) # convert back to list to index
                    print(f"Drop {dropped_vars_list[0]}, {dropped_vars_list[1]},  and {dropped_vars_list[2]}:")
                    # find log likelihood comparison for likelihood ratio test
                    # comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
                    LRT = 2 * ((-model_drop.llf) - (-log_reg.llf)) 
                    # LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
                    # survival function is (1 - cdf) for LRT value given degrees of freedom  
                    p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop([dropped_vars_list[0], dropped_vars_list[1], dropped_vars_list[2]], axis=1).shape[1])

                    # save model and info to list 
                    description_string = f"{idx_count} - Dropped {dropped_vars_list[0]}, {dropped_vars_list[1]}, and {dropped_vars_list[2]} from full model"
                    if dropped_vars in dropped_vars_dicts_list:
                        idx_count-= 1
                        continue
                    else:
                        dropped_models.append(model_drop)
                        dropped_models_info.append(description_string)
                        dropped_models_LRT.append(p_value)
                        dropped_vars_dicts_list.append(dropped_vars)

                    # print(model_drop.summary())   
                    # print(f"LRT statistic: {LRT:.4f}")
                    # print(f"P-value: {p_value:.4f}\n")


    return log_reg, dropped_models, dropped_models_info, dropped_models_LRT


def predict_and_visualise(sim_data, full_model, predict_vars, bin_max, bin_min=0, dropped_models=None, dropped_models_idx=None):
    """ Using generated logistic regression models, predict on simulated data and visualise
        If no dropped_models parameters given, will use full_model for predictions
        Inputs:
         sim_data - simulated data in the form of a dataframe
         full_model - full log reg model
         predict_vars - (column) names of variables to predict on
         bin_max, bin_min - heatmap bin limits (for x and y, not z)
         dropped_models - list of models with dropped terms
         dropped_models_idx - index of the dropped_model to be used

         Output:
         Display heatmap of specified model predictions on simulated data
           """

    # use specified model to predict data and display output as a heatmap 
    if not dropped_models:
        # full model
        x_test = predict_vars
        x_test = sim_data[x_test]
        y_test = full_model.predict(x_test)
        
        # feed chose_high and data in heatmaps
        heatmap(sim_data.head_angles_wall_1.values, sim_data.head_angles_wall_2.values, y_test, 5,
                bin_max=bin_max, x_title='Head angle to High (°)', y_title='Head angle to Low (°)', cut_low_n=True)
        
    else:
        # dropped model
        x_test = predict_vars
        x_test = sim_data[x_test]
        y_test = dropped_models[dropped_models_idx].predict(x_test)
        
        # feed chose_high and data in heatmaps
        heatmap(sim_data.head_angles_wall_1.values, sim_data.head_angles_wall_2.values, y_test, 5,
                bin_max=bin_max, x_title='Head angle to High (°)', y_title='Head angle to Low (°)', cut_low_n=True)
        
        return None



# BODY

### CREATE SIM DATA ### 

### set limits for data values ###
head_ang_min, head_ang_max = 0, math.pi
distance_wall_min, distance_wall_max = 0, 2

### randomise 10000 trials of distance and head_angle ###
distances_wall_1 = np.random.rand(ITERATIONS) * distance_wall_max
distances_wall_2 = np.random.rand(ITERATIONS) * distance_wall_max

head_angles_wall_1 = np.random.rand(ITERATIONS) * head_ang_max
head_angles_wall_2 = np.random.rand(ITERATIONS) * head_ang_max

bias_term = np.ones(distances_wall_1.shape[0])

data = {'bias_term':bias_term,
        'distances_wall_1':distances_wall_1, 'distances_wall_2':distances_wall_2,
        'head_angles_wall_1':np.degrees(head_angles_wall_1), 'head_angles_wall_2':np.degrees(head_angles_wall_2)}
sim_data = pd.DataFrame(data)


### CREATE LOGISTIC REGRESSION MODELS ###

# PARAMS
if SOCIAL:
    project_type = 'octagon_multi'
else:
    project_type = 'octagon_solo'


# create session objects
root = '/home/tomhagley/Documents/SLEAPProject'
sessions = ['2022-11-04_A004', '2022-11-02_A004', '2022-10-31_A004', '2022-10-28_A004']
sessions_list = [Session(root, session) for session in sessions]

# extract data for all sessions
for session in sessions_list:
    session.data_filtering('all')
    session.extract_basic_data()

    
# identify whether chose high
# remember to define whether grating is high
chose_high = np.concatenate([session.octpy_metadata.chose_light for session in sessions_list])
if GRATING_HIGH:
    chose_high = np.where(chose_high==True, False, True).astype('int') 
else:
    chose_high = np.where(chose_high==True, True, False).astype('int')

# concatenate multiple variables across sessions
# define variables you want concatenated
vars = ["distances_wall_1", "distances_wall_2", "head_angles_wall_1", "head_angles_wall_2"]

# create dataframe
df = pd.DataFrame()
# loop through var names and concate them, then add to df 
for var in vars:
    df[f"{var}"] = np.concatenate([getattr(session, f"{var}") for session in sessions_list])

df['chose_high'] = chose_high

bias_term = np.ones(df['distances_wall_1'].shape[0])
df['bias_term'] = bias_term

# create dataframe of all variables
# convert radians to degrees for angle terms
data_input = {'chose_high':chose_high.astype('int'), 'bias_term':bias_term, 
        'distances_wall_1':df.distances_wall_1.values, 'distances_wall_2':df.distances_wall_2.values,
        'head_angles_wall_1':np.degrees(df.head_angles_wall_1.values), 'head_angles_wall_2':np.degrees(df.head_angles_wall_2.values)}

# list of all variables eligible for dropping
drop_vars = ['distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']
# list of all independent variables
IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']



# run model training on experimental data
# provide experimental data input, independent variable column names (to use in training),
# dependent variable column name, and column names of variables to drop
(full_model, dropped_models, 
 dropped_models_desc, dropped_models_LRT) = train(data_input, IVs, 'chose_high', drop_vars)


########### Interesting head-angle

# model trained on all terms, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0)

# model trained on only head_angles_wall_1, head_angles_wall_2, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'head_angles_wall_1', 'head_angles_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=1)

# model trained on distances only, predict chose_high on sim data
# (should get nothing here, becuase the model has no information about head angle, and is trying to predict
#  choose_high at binned distances only using distance coefficients)
prediction_IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=12)

# model trained on all but head_angles_wall_1, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angles_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=11)

# model trained on all but head_angles_wall_2, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angles_wall_1']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=13)

# model trained on only head_angles_wall_1, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'head_angles_wall_1']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=5)

# model trained on only head_angles_wall_2, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'head_angles_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=4)

################ Superfluous

# model trained on all but distances_wall_1, predict chose_high on sim data
prediction_IVs = ['bias_term', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=0)

# model trained on all but distances_wall_2, predict chose_high on sim data
prediction_IVs = ['bias_term', 'distances_wall_1', 'head_angles_wall_1', 'head_angles_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=7)

# model trained on only distances_wall_2, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'distances_wall_2']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=6)

# model trained on only distances_wall_1, predicting chose_high on sim data
prediction_IVs = ['bias_term', 'distances_wall_1']
predict_and_visualise(sim_data, full_model, prediction_IVs, BIN_MAX, 0, dropped_models=dropped_models, dropped_models_idx=10)



if PRINT_ALL:
    for i in range(len(dropped_models)):
        print(dropped_models_desc[i])
        print(dropped_models[i].summary())
        print(f"p value: {dropped_models_LRT[i]:.2f}")