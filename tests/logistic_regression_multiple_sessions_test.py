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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from session import Session



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
vars = ["distances_wall_1", "distances_wall_2", "head_angles_wall_1", "head_angles_wall_2"]

# create dataframe
df = pd.DataFrame()
# loop through var names and concate them, then add to df 
for var in vars:
    df[f"{var}"] = np.concatenate([getattr(session, f"{var}") for session in sessions_list])

df['chose_high'] = chose_high

bias_term = np.ones(df['distances_wall_1'].shape[0])
df['bias_term'] = bias_term

description = "model with distances and head angles to both walls"

# define independent variables:
independent_variables = np.hstack([df.distances_wall_1.values.reshape(-1,1), df.distances_wall_2.values.reshape(-1,1), 
                                    df.head_angles_wall_1.values.reshape(-1,1), df.head_angles_wall_2.values.reshape(-1,1)])




print('### SKLEARN LOGISTIC REGRESSION ###')
model_full = logistic_regression_choose_high(chose_high, independent_variables, df.head_angles_wall_2.values.reshape(-1,1),
                                            description_string=description)   

# do statsmodels version to find p-values
# create dataframe of variables
data = {'chose_high':chose_high.astype('int'), 'bias_term':bias_term, 
        'distances_wall_1':df.distances_wall_1.values, 'distances_wall_2':df.distances_wall_2.values,
        'head_angles_wall_1':df.head_angles_wall_1.values, 'head_angles_wall_2':df.head_angles_wall_2.values}

var_names = ['distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']

data = pd.DataFrame(data)
Xtrain = data[['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']]
ytrain = data['chose_high']

print('\n\n### STATSMODELS LOGISTIC REGRESSION ###')
log_reg = sm.Logit(ytrain, Xtrain).fit()
print()
print(log_reg.summary())

log_reg_2 = sm.Logit.from_formula('chose_high ~ distances_wall_1 + distances_wall_2' +
                                    ' + head_angles_wall_1 + head_angles_wall_2', data).fit()
# This model comes out identical to the first
# print(log_reg_2.summary())
# print(log_reg_2.summary2())

####################### dropping each of the 4 terms 
IV = data.drop('chose_high', axis=1)
IV = IV.drop('bias_term', axis=1)
# # perform drop1 analysis
# for var in var_names:
#     model_drop = sm.Logit.from_formula(f'chose_high ~ {"bias_term + " + " + ".join(IV.columns.drop(var))}', data).fit()
#     print(f"Drop {var}:")
#     print(model_drop.summary())
#     # find log likelihood comparison for likelihood ratio test
#     # comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
#     LRT = 2 * ((-model_drop.llf) - (-log_reg_2.llf)) 
#     # LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
#     # survival function is (1 - cdf) for LRT value given degrees of freedom  
#     p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop(var, axis=1).shape[1])

#     print(f"LRT statistic: {LRT:.4f}")
#     print(f"P-value: {p_value:.4f}\n")

############ COMPARISON WITH DROPPED MODELS
# drop each variable in turn and compare it to full model
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
    LRT = 2 * ((-model_drop.llf) - (-log_reg_2.llf)) 
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
print(f"Drop distances_wall_1, distances_wall_2:")
print(model_drop.summary())
# find log likelihood comparison for likelihood ratio test
# comparison = 2 * (-ln(Lsimple)) - (-ln(Lcomplex))
LRT = 2 * ((-model_drop.llf) - (-log_reg_2.llf)) 
# LRT statistic roughly follows chi squared dist., DoF are the number of additional terms
# survival function is (1 - cdf) for LRT value given degrees of freedom  
p_value = scipy.stats.chi2.sf(LRT, df=IV.shape[1] - IV.drop(var, axis=1).shape[1])

print(f"LRT statistic: {LRT:.4f}")
print(f"P-value: {p_value:.4f}\n")

