from utils.plotting.heatmap import heatmap
import numpy as np
import pandas as pd
from session import Session


GRATING_HIGH = False
BIN_MAX = 2.01


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
sessions = ['2023-02-27_ADU-M-0003', '2023-02-24_ADU-M-0003', '2023-02-21_ADU-M-0003', '2023-02-19_ADU-M-0003']
sessions_list = [Session(root, session) for session in sessions]

# extract data for all sessions
for session in sessions_list:
    session.data_filtering('all')
    session.extract_basic_data()



# get multiple variables
vars = ["distances_wall_1", "distances_wall_2", "head_angles_wall_1", "head_angles_wall_2"]
# create dataframe
df = pd.DataFrame()
for var in vars:
    df[f"{var}"] = np.concatenate([getattr(session, f"{var}") for session in sessions_list])


# get specific single variables
# change chose_high depending on whether grating is high!
chose_high = np.concatenate([session.octpy_metadata.chose_light for session in sessions_list])
if GRATING_HIGH == True:
    chose_high = np.where(chose_high==True, False, True).astype('int')
elif GRATING_HIGH == False:
    chose_high = np.where(chose_high==True, True, False).astype('int')
df['chose_high'] = chose_high

# plot heatmap
heatmap(df.distances_wall_1.values, df.distances_wall_2.values, df.chose_high.values, 5,
        bin_max=BIN_MAX, x_title='Distance to High', y_title='Distance to Low', cut_low_n=True)