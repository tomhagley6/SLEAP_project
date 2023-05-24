import numpy as np
import pandas as pd
from session import Session

""" Load multiple sessions, then concatenate chosen data and store in DataFrame """

# create session objects
root = '/home/tomhagley/Documents/SLEAPProject'
sessions = ['2022-11-04_A004', '2022-11-02_A004', '2022-10-31_A004', '2022-10-28_A004']
sessions_list = [Session(root, session) for session in sessions]

# extract data for all sessions
for session in sessions_list:
    session.data_filtering('all')
    session.extract_basic_data()

# concatenate single specific variable
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
