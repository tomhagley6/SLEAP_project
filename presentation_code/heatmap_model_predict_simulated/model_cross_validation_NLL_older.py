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

""" Script to generate logistic regression models on training dataset
    Then predict on test dataset and calculate the negative log-likelihood
     
    Example NLLs and mean_diffs with the original 8 model choices (first is full model) and the A004 dataset:
     NLLs = [46.78653438984054, 55.78840895784988, 53.516841506447385, 60.313311287583524, 58.91318398912639, 55.07415860394206, 61.80344442870653, 69.96287000963532]
     mean_diffs = [0.19787981961193724, 0.22608346927889486, 0.245031720619203, 0.23668003623664216, 0.23196554459980523, 0.24638394729901358, 0.2405106142288918, 0.29453297546846635]
     """


# GLOBALS
ITERATIONS = 10000          # simulated data iters
SOCIAL = False              # social session
GRATING_HIGH = True         # is grating High reward
PRINT_ALL = False           # print summaries of all models
BIN_MAX = 2.01              # x,y max bin values for the heatmap

EXAMPLE_NLLS = [46.78653438984054, 55.78840895784988, 
                53.516841506447385, 60.313311287583524, 58.91318398912639, 55.07415860394206, 61.80344442870653, 69.96287000963532]
EXAMPLE_MEAN_DIFFS = [0.19787981961193724, 0.22608346927889486, 0.245031720619203, 0.23668003623664216, 0.23196554459980523,
                       0.24638394729901358, 0.2405106142288918, 0.29453297546846635]

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


def predict(sim_data, full_model, predict_vars, bin_max, bin_min=0, dropped_models=None, dropped_models_idx=None):
    """ Using generated logistic regression models, predict on simulated data and visualise
        If no dropped_models parameters given, will use full_model for predictions
        Inputs:
         sim_data - simulated data in the form of a dataframe
         full_model - full log reg model
               
        # # feed chose_high and data in heatmaps
        # heatmap(sim_data.distances_wall_1.values, sim_data.distances_wall_2.values, y_test, 5,
        #         bin_max=bin_max, x_title='Distance to High', y_title='Distance to Low', cut_low_n=True)  predict_vars - (column) names of variables to predict on
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
        
        # # feed chose_high and data in heatmaps
        # heatmap(sim_data.distances_wall_1.values, sim_data.distances_wall_2.values, y_test, 5,
        #         bin_max=bin_max, x_title='Distance to High', y_title='Distance to Low', cut_low_n=True)
        
    else:
        # dropped model
        x_test = predict_vars
        x_test = sim_data[x_test]
        y_test = dropped_models[dropped_models_idx].predict(x_test)
        
        # # feed chose_high and data in heatmaps
        # heatmap(sim_data.distances_wall_1.values, sim_data.distances_wall_2.values, y_test, 5,
        #         bin_max=bin_max, x_title='Distance to High', y_title='Distance to Low', cut_low_n=True)
        
    
    return y_test



# BODY


### CREATE LOGISTIC REGRESSION MODELS ###

# PARAMS
if SOCIAL:
    project_type = 'octagon_multi'
else:
    project_type = 'octagon_solo'


# create session objects
root = '/home/tomhagley/Documents/SLEAPProject'
sessions = ['2022-11-04_A004', '2022-11-02_A004', '2022-10-31_A004', '2022-10-28_A004']
# sessions = ['2023-02-27_ADU-M-0003', '2023-02-24_ADU-M-0003', '2023-02-21_ADU-M-0003', '2023-02-19_ADU-M-0003']
# sessions = ['2022-12-20_ADU-M-0003', '2022-12-19_ADU-M-0003', '2022-12-14_ADU-M-0003', '2022-12-14_ADU-M-0003']

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


### Separate into train and test sets
# create array of same length to generate random indexes from 
idx_array = np.arange(df.chose_high.values.size)

# sample 80% of idxs randomly for training data
idxs_training = np.random.choice(idx_array, int(idx_array.size*0.8), replace=False)
idxs_test = np.array([i for i in range(df.chose_high.values.size) if i not in idxs_training])

# use these arrays to sample the dataframe
df_train = pd.DataFrame()
for column in df:
    df_train[f"{column}"] = df[f"{column}"][list(idxs_training)].values
df_test = pd.DataFrame()
for column in df:
    df_test[f"{column}"] = df[f"{column}"][list(idxs_test)].values

# format training data for training
data_input = {'chose_high':df_train.chose_high.astype('int'), 'bias_term':df_train.bias_term, 
        'distances_wall_1':df_train.distances_wall_1.values, 'distances_wall_2':df_train.distances_wall_2.values,
        'head_angles_wall_1':df_train.head_angles_wall_1.values, 'head_angles_wall_2':df_train.head_angles_wall_2.values}

# format test data for prediction
test_data = {'bias_term':df_test.bias_term, 
        'distances_wall_1':df_test.distances_wall_1.values, 'distances_wall_2':df_test.distances_wall_2.values,
        'head_angles_wall_1':df_test.head_angles_wall_1.values, 'head_angles_wall_2':df_test.head_angles_wall_2.values}
test_data = pd.DataFrame(test_data)

# list of all variables eligible for dropping
drop_vars = ['distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']
# list of all independent variables
IVs = ['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']
# list of aldata_inputl variables to predict on
predict_vars = ['bias_term', 'distances_wall_1', 'distances_wall_2', 'head_angles_wall_1', 'head_angles_wall_2']

# run model training on experimental data
# provide experimental data input, independent variable column names (to use in training),
# dependent variable column name, and column names of variables to drop
(full_model, dropped_models, 
 dropped_models_desc, dropped_models_LRT) = train(data_input, IVs, 'chose_high', drop_vars)

### predict on experimental data in the test dataset
### do this for full model, and for relevant dropped models
### currently 8 models total
y_hats = []
ys = []
# dropped: distances_wall_1
#           distances_wall_2
#           head_angles_wall_1
#           head_angles_wall_2
#           distances_wall_1, distances_wall_2
#           head_angles_wall_1, head_angles_wall_2
#           head_angles_wall_2, distances_wall_1, distances_wall_2
dropped_models_idxs = [0, 7, 11, 13, 1, 12, 5] # 7 dropped models total 

# predict on full model
y_hat = predict(test_data, full_model, predict_vars, BIN_MAX, bin_min=0, dropped_models=None, dropped_models_idx=None)
y = df_test.chose_high
y_hats.append(y_hat)

# loop through dropped models
for idx in dropped_models_idxs:
    # first find the variables that the dropped model is trained on 
    predict_vars = list(dropped_models[idx].params.index.values)
    y_hat = predict(test_data, full_model, predict_vars, BIN_MAX, bin_min=0, dropped_models=dropped_models, dropped_models_idx=idx)
    y_hats.append(y_hat)

# use the NLL equation to evaluate each model on the test data
#  -(log(y_hat)*y + log(1 - y_hat)*(1-y))
# where y_hat is prediction value, y is binary ground truth
NLLs = []
for i in range(len(y_hats)):
    NLL = -(np.log(y_hats[i])*y + np.log(1 - y_hats[i])*(1-y))
    NLL = NLL.sum()
    NLLs.append(NLL)

print(NLLs)

# find the mean aboslute difference between prediction and ground truth for all models
mean_diffs = []
for i in range(len(y_hats)):
    mean_diff = np.mean(abs(y_hats[i] - y))
    mean_diffs.append(mean_diff)
# plot the results for major model types
    # TODO







