from utils.plotting.heatmap import heatmap
import numpy as np
import pandas as pd
from session import Session
import math

""" Script to plot a heatmap of head angles at the start of trial,
    across x and y position in the octagon """

GRATING_HIGH = False

# ### SINGLE SESSION ### 
# # setup session instance
# root = '/home/tomhagley/Documents/SLEAPProject'
# session = '2022-11-04_A004'
# sess = Session(root, session)
# sess.extract_basic_data()

# # define variables
# chose_high = sess.octpy_metadata.chose_light
# chose_high = np.where(chose_high==True, False, True).astype('int')

# # plot heatmap 
# heatmap(sess.distances_wall1, sess.distances_wall2, chose_high, 4, 
#         x_title='distance to high', y_title='distance to low')

### MULTIPLE SESSIONS ###
# create session objects
root = '/home/tomhagley/Documents/SLEAPProject'
sessions = ['2022-12-20_ADU-M-0003', '2022-12-19_ADU-M-0003', '2022-12-14_ADU-M-0003', '2022-12-14_ADU-M-0003'] # early learning
sessions = sessions + ['2023-02-27_ADU-M-0003', '2023-02-24_ADU-M-0003', '2023-02-21_ADU-M-0003', '2023-02-19_ADU-M-0003'] # late learning
sessions_list = [Session(root, session) for session in sessions]

# extract data for all sessions
for session in sessions_list:
    session.data_filtering('all')
    session.extract_basic_data()



# get multiple variables
vars = ["trajectories", "head_angles"]
# create dataframe
df = pd.DataFrame()

# get the bodyupper x and y coordinates from trajectories for all concatenated sessions
bodyupper_x = pd.concat([(getattr(session, f"{vars[0]}").bodyupper_x) for session in sessions_list], axis=0)
bodyupper_y = pd.concat([(getattr(session, f"{vars[0]}").bodyupper_y) for session in sessions_list], axis=0)

# pandas allows using the string accessor for np arrays - use this to get the locations at the first
# timepoint of each trial
# also invert y values because of -1 being top
df["start_x"], df["start_y"] = bodyupper_x.str[0], bodyupper_y.str[0] * -1

# also get head angle at start timepoint of each trial
df[f"{vars[1]}"] = np.concatenate([getattr(session, f"{vars[1]}") for session in sessions_list])


# # plot heatmap of mean head angle at trial start (z) for x and y start locations (x,y)
heatmap(df.start_x.values, df.start_y.values, df.head_angles.values, 5,
        bin_max=1.01, bin_min=-1.01, v_max = 2*math.pi, v_min=0, x_title='X Location', y_title='Y Location', cut_low_n=True)