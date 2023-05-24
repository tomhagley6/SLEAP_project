from utils.plotting.heatmap import heatmap
import numpy as np
import pandas as pd
from session import Session
import math
from head_angle import extract_head_angle_frame_session
from utils.get_values.get_node_numbers import get_node_numbers


""" Script to plot a heatmap of head angles at the start of trial,
    across x and y position in the octagon """

GRATING_HIGH = False

### MULTIPLE SESSIONS ###
# create session objects
root = '/home/tomhagley/Documents/SLEAPProject'
# sessions = ['2022-12-20_ADU-M-0003']
sessions = ['2022-12-20_ADU-M-0003', '2022-12-19_ADU-M-0003', '2022-12-14_ADU-M-0003', '2022-12-14_ADU-M-0003'] # early learning
# sessions = sessions + ['2023-02-27_ADU-M-0003', '2023-02-24_ADU-M-0003', '2023-02-21_ADU-M-0003', '2023-02-19_ADU-M-0003'] # late learning
sessions_list = [Session(root, session) for session in sessions]

# extract data for all sessions
for session in sessions_list:
    session.data_filtering('all')
    session.extract_basic_data()
    session.more_complex_extraction()

    # find head angle at each frame
    head_angle_frames_session = extract_head_angle_frame_session(session.labels_file,
                                                                 track_num=0)
    session.head_angle_frames_session = head_angle_frames_session



# get multiple variables
vars = ["video_trajectories", "head_angle_frames_session"]
# create dataframe
df = pd.DataFrame()

# get the bodyupper x and y coordinates from trajectories for all frames in concatenated sessions
# first find the indices for the correct node of the skeleton 
node_num = get_node_numbers('bodyupper')
node_idx_x = node_num*2
node_idx_y = node_num*2 + 1
bodyupper_x = np.concatenate([(getattr(session, f"{vars[0]}")[:,node_idx_x,0]) for session in sessions_list], axis=0)
bodyupper_y = np.concatenate([(getattr(session, f"{vars[0]}")[:, node_idx_y,0]) for session in sessions_list], axis=0)

# invert y values because of -1 being top
df["x_coord"], df["y_coord"] = bodyupper_x, bodyupper_y * -1

# also get head angle at each frame of the session
df[f"{vars[1]}"] = np.concatenate([getattr(session, f"{vars[1]}") for session in sessions_list])


# # plot heatmap heatmap of mean head angle at trial start (z) for x and y start locations (x,y)
heatmap(df.x_coord.values, df.y_coord.values, df.head_angle_frames_session.values, 5,
        bin_max=1.00, bin_min=-1.00, v_max = 2*math.pi, v_min=0, x_title='X Location', y_title='Y Location', cut_low_n=True)