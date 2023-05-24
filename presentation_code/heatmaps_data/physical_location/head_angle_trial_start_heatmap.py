from utils.plotting.heatmap import heatmap
import numpy as np
import pandas as pd
from session import Session
import math

""" Script to plot a heatmap of head angles at the start of trial,
    across x and y position in the octagon """

GRATING_HIGH = False

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


# # plot heatmap heatmap of mean head angle at trial start (z) for x and y start locations (x,y)
heatmap(df.start_x.values, df.start_y.values, np.degrees(df.head_angles.values), 5,
        bin_max=1.01, bin_min=-1.01, v_max = 360, v_min=0, x_title='X Location', y_title='Y Location', cut_low_n=True)