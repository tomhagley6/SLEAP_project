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


ITERATIONS = 10000 


### CREATE SIM DATA ### 

### set limits for data values ###
head_ang_min, head_ang_max = 0, math.pi
distance_wall_min, distance_wall_max = 0, 2

### randomise 10000 trials of distance and head_angle ###
distances_wall_1 = np.random.rand(ITERATIONS) * distance_wall_max
distances_wall_2 = np.random.rand(ITERATIONS) * distance_wall_max

head_angle_wall_1 = np.random.rand(ITERATIONS) * head_ang_max
head_angle_wall_2 = np.random.rand(ITERATIONS) * head_ang_max

bias_term = np.ones(distances_wall_1.shape[0])

data = {'bias_term':bias_term,
        'distances_wall_1':distances_wall_1, 'distances_wall_2':distances_wall_2,
        'head_angle_wall_1':head_angle_wall_1, 'head_angle_wall_2':head_angle_wall_2}
sim_data = pd.DataFrame(data)


### CREATE LOGISTIC REGRESSION MODELS ###

# PARAMS
SOCIAL = False
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
# currently assuming grating is high!
chose_high = np.concatenate([session.octpy_metadata.chose_light for session in sessions_list])
chose_high = np.where(chose_high==True, False, True).astype('int')

# concatenate multiple variables
# define variables you want concatenated
vars = ["distance_wall_1", "distance_wall_2", "head_angle_wall_1", "head_angle_wall_2"]

# create dataframe
df = pd.DataFrame()
# loop through var names and concate them, then add to df 
for var in vars:
    df[f"{var}"] = np.concatenate([getattr(session, f"{var}") for session in sessions_list])

df['chose_high'] = chose_high

bias_term = np.ones(df['distance_wall_1'].shape[0])
df['bias_term'] = bias_term

description = "model with distances and head angles to both walls"

# create dataframe of variables
data_input = {'chose_high':chose_high.astype('int'), 'bias_term':bias_term, 
        'distances_wall_1':df.distance_wall_1.values, 'distances_wall_2':df.distance_wall_2.values,
        'head_angle_wall_1':df.head_angle_wall_1.values, 'head_angle_wall_2':df.head_angle_wall_2.values}

var_names = ['distances_wall_1', 'distances_wall_2', 'head_angle_wall_1', 'head_angle_wall_2']

# define Xtrain (IVs) and ytrain (DV)
data = pd.DataFrame(data_input)
# Xtrain = data[['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_to_wall_1_angs', 'head_to_wall_2_angs']]
Xtrain = data[['bias_term', 'distances_wall_1', 'distances_wall_2']]
ytrain = data['chose_high']

# print summary
print('\n\n### STATSMODELS LOGISTIC REGRESSION ###')
log_reg = sm.Logit(ytrain, Xtrain).fit()
print(log_reg.summary())
print(log_reg.summary2())

# drop 
IV = data.drop('chose_high', axis=1)
IV = IV.drop('bias_term', axis=1)

dropped_models = []
for var in var_names:
    Xtrain = [x for x in data.columns if x is not f"{var}" and x is not 'chose_high' and x is not 'bias_term']
    Xtrain = data[Xtrain]
    ytrain = data['chose_high']
    model_drop = sm.Logit(ytrain, Xtrain).fit()
    print(f"Drop {var}:")
    print(model_drop.summary())
    # find log likelihood comparison for likelihood ratio test
    # comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
    LRT = 2 * ((-model_drop.llf) - (-log_reg.llf)) 
    # LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
    # survival function is (1 - cdf) for LRT value given degrees of freedom  
    p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop(var, axis=1).shape[1])

    # save model to list 
    dropped_models.append(model_drop)

    print(f"LRT statistic: {LRT:.4f}")
    print(f"P-value: {p_value:.4f}\n")

# drop both distance variables and compare it to full model
Xtrain = [x for x in data.columns if x not in ['chose_high', 'distances_wall_1', 'distances_wall_2']]
Xtrain = data[Xtrain]
ytrain = data['chose_high']
model_drop = sm.Logit(ytrain, Xtrain).fit()
print(f"Drop distance_wall_1, distance_wall_2:")
print(model_drop.summary())
# find log likelihood comparison for likelihood ratio test
# comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
LRT = 2 * ((-model_drop.llf) - (-log_reg.llf)) 
# LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
# survival function is (1 - cdf) for LRT value given degrees of freedom  
p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop(var, axis=1).shape[1])

# save model to list
dropped_models.append(model_drop)

print(f"LRT statistic: {LRT:.4f}")
print(f"P-value: {p_value:.4f}\n")

### PREDICT USING LOGISTIC REGRESSION MODELS ### 


# full model
x_test = [x for x in sim_data.columns if x is not 'head_angle_wall_1' and x is not 'head_angle_wall_2']
x_test = sim_data[x_test]
y_test = log_reg.predict(x_test)



# feed chose_high and data in heatmaps
heatmap(sim_data.distances_wall_1.values, sim_data.distances_wall_2.values, y_test, 5,
        bin_max=2.01, x_title='distance to High', y_title='distance to Low', cut_low_n=True)



def train(data, xvars, yvar, drop_vars, predict_vars):
    """ Train full and dropped-term logistic regression models
      
       Input:
        data - dictionary of column_name:column_data as type string:np.ndarray
        xvars - list of strings of independent variable column_names to train on
        yvar - string of dependent variable column_name
        drop_vars - list of strings of variables to drop sequentially
        predict_vars - list of strings of variables to use in predicting
        on simulated data
        
        Output:
         full model and list of dropped models with descriptions"""
    
    # define Xtrain (IVs) and ytrain (DV)
    data = pd.DataFrame(data)

    # Xtrain = data[['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_to_wall_1_angs', 'head_to_wall_2_angs']]
    # only train the model with the variables used to predict
    Xtrain = data[predict_vars]
    ytrain = data[yvar]

    # print summary
    print('\n\n### STATSMODELS LOGISTIC REGRESSION ###')
    log_reg = sm.Logit(ytrain, Xtrain).fit()
    print(log_reg.summary())
    print(log_reg.summary2())

    # drop single terms from the model 
    IV = data.drop('chose_high', axis=1)
    IV = IV.drop('bias_term', axis=1)

    # save the model, a short description, and p-value of LLR comparison to full model
    dropped_models = []
    dropped_model_info = []
    dropped_models_LRT = []
    for var in drop_vars:
        Xtrain = [x for x in data.columns if x is not f"{var}" and x is not 'chose_high' and x is not 'bias_term']
        Xtrain = data[Xtrain]
        ytrain = data['chose_high']
        model_drop = sm.Logit(ytrain, Xtrain).fit()
        print(f"Drop {var}:")
        print(model_drop.summary())
        # find log likelihood comparison for likelihood ratio test
        # comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
        LRT = 2 * ((-model_drop.llf) - (-log_reg.llf)) 
        # LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
        # survival function is (1 - cdf) for LRT value given degrees of freedom  
        p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop(var, axis=1).shape[1])

        # save model to list 
        dropped_models.append(model_drop)
        dropped_model_info.append(f"Dropped {var} from full model")
        dropped_models_LRT.append(p_value)

        print(f"LRT statistic: {LRT:.4f}")
        print(f"P-value: {p_value:.4f}\n")

    # drop both distance variables and compare it to full model
    Xtrain = [x for x in data.columns if x not in ['chose_high', 'distances_wall_1', 'distances_wall_2']]
    Xtrain = data[Xtrain]
    ytrain = data['chose_high']
    model_drop = sm.Logit(ytrain, Xtrain).fit()
    print(f"Drop distance_wall_1, distance_wall_2:")
    print(model_drop.summary())
    # find log likelihood comparison for likelihood ratio test
    # comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
    LRT = 2 * ((-model_drop.llf) - (-log_reg.llf)) 
    # LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
    # survival function is (1 - cdf) for LRT value given degrees of freedom  
    p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop(var, axis=1).shape[1])

    # save dropped-distance model to list
    dropped_models.append(model_drop)
    dropped_models_LRT.append(p_value)
    dropped_model_info.append("Dropped distance_wall_1 and distance_wall_2 from full model")

    print(f"LRT statistic: {LRT:.4f}")
    print(f"P-value: {p_value:.4f}\n")

    return log_reg, dropped_models, dropped_model_info, dropped_models_LRT

def predict_and_visualise(sim_data, full_model, predict_vars, bin_max, bin_min=0, dropped_models=None, dropped_models_idx=None):
    """ Using generated logistic regression models, predict on simulated data and visualise
        Inputs:
         sim_data - simulated data in the form of a dataframe
         full_model - full log reg model
         predict_vars - (column) names of variables to predict on
         bin_max, bin_min - heatmap bin limits
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
        heatmap(sim_data.distances_wall_1.values, sim_data.distances_wall_2.values, y_test, 5,
                bin_max=bin_max, x_title='distance to High', y_title='distance to Low', cut_low_n=True)
        
    else:
        # dropped model
        x_test = predict_vars
        x_test = sim_data[x_test]
        y_test = dropped_models[dropped_models_idx].predict(x_test)
        
        # feed chose_high and data in heatmaps
        heatmap(sim_data.distances_wall_1.values, sim_data.distances_wall_2.values, y_test, 5,
                bin_max=bin_max, x_title='distance to High', y_title='distance to Low', cut_low_n=True)
        
        return None

### BODY

IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angle_wall_1', 'head_angle_wall_2']
distance_IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2']

# run model training on experimental data
(full_model, dropped_models, 
 dropped_model_desc, dropped_models_LRT) = train(data_input, var_names, 'chose_high', var_names, IVs )

predict_and_visualise(sim_data, full_model, IVs, 1, 0 )


for i in range(len(dropped_models)):
    print(dropped_model_desc[i])
    print(dropped_models[i].summary())
    print(f"p value: {dropped_models_LRT[i]:.2f}")