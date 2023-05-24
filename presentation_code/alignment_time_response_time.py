import os
import pandas as pd
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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

""" load in necessary data from session, fit linear regression model
    plot model alongside data as 'RT' against 'time to align head to walls' """

# PARAMS
SOCIAL = False
if SOCIAL:
    project_type = 'octagon_multi'
else:
    project_type = 'octagon_solo'

session = '2022-11-04_A004'
node_name = 'BodyUpper' # general node to use

# PATHS
# trial metadata
data_root = '/home/tomhagley/Documents/SLEAPProject/data'

# greyscale video
video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/videos/{session}'
video_filename = f'CameraTop_{session}' + '_full.avi'
video_path_greyscale = video_root + os.sep + video_filename

# color video 
color_video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/videos/{session}'
color_video_filename = f'CameraColorTop_{session}' + '_full.avi'
video_path = color_video_root + os.sep + color_video_filename

# tracking data
trajectories_directory = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/exports'
# trajectories_filename = 'CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
trajectories_filename = 'CameraTop_2022-11-04_A004_full_model9_predictions_CLI.analysis.h5'

trajectories_filepath = trajectories_directory + os.sep + trajectories_filename

# labels file
labels_directory = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/predictions/'
# labels_filename = 'CameraTop_2022-11-02_A006_full_predictions_model5_CLI.slp'
labels_filename = 'CameraTop_2022-11-04_A004_full_model9_predictions_CLI.slp'
labels_filepath = labels_directory + os.sep + labels_filename
labels_file = sleap.load_file(labels_filepath)

# video saving
save_video_output_path = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/analysis'

# load metadata files from HPC using octpy
octpy_metadata, video_metadata, color_video_metadata = find_frames.access_metadata(data_root, session, 
                                                                                   colorVideo=True, refreshFiles=False)

# filter data if required (e.g. RT < 15 s, choice-trials only)
octpy_metadata = data_filtering.choice_trials_only(octpy_metadata)

# find ColorCamera and GreyscaleCamera frame numbers for trial start, stim start, and trial end
grey_trials, grey_stims, grey_ends, \
color_trials, color_stims, color_ends = find_frames.relevant_session_frames(octpy_metadata, 
                                                                                            video_metadata, 
                                                                                            color_video_metadata)


# BODY
### winner mouse track ###
if SOCIAL:
    # TODO find which track contains the winner mouse for each trial and assign to self
    # create a tracks array that records track of winner for each trial
    pass
else:
    tracks = np.zeros(octpy_metadata.shape[0]).astype('int')

### TRAJECTORIES ###

### video trajectory ###
# find continuous trajectory across full video
video_trajectories, col_names = trajectory_extraction.extract_video_trajectory(trajectories_filepath, normalise=True,
                                                                    smooth=True)

### head angle to high and low wall ### 
head_angles_wall_1, head_angles_wall_2 = get_head_to_wall_angle_session(octpy_metadata, video_trajectories,
                                                                               grey_stims, tracks, labels_file)


#  4. time to align head angle with walls
# first find the head_to_wall_ang for every frame in every trial in session
head_wall_1_ang_all_frames, head_wall_2_ang_all_frames = get_head_to_wall_angle_full_trial_full_session(octpy_metadata, video_trajectories, 
                                                                               grey_stims, grey_ends, tracks, labels_file, grating_high=True)
# now find the time to align with either of the two walls (angle specificity is in function file)
times_to_head_wall_alignment = time_to_alignment_session(head_wall_1_ang_all_frames, head_wall_2_ang_all_frames)
# find the trial response times to compare with
response_times_sess = response_times_session(octpy_metadata)

# do linear regression
x = np.array(times_to_head_wall_alignment).reshape(-1,1)
y = response_times_sess
model = LinearRegression().fit(x,y)
c = model.intercept_
m = model.coef_
r_sq = model.score(x,y)

def f(x, m, c):
    return m*x + c
x = np.linspace(0,3.1,100)

sns.set(font_scale=1.2)
sns.set_style("darkgrid", {"xtick.bottom":True, "ytick.left":True})
# sns.set_style("ticks")
df = pd.DataFrame({'times_to_head_wall_alignment':times_to_head_wall_alignment, 'response_times_sess':response_times_sess})
fig, p1 = plt.subplots()
p1 = sns.scatterplot(data=df, x='times_to_head_wall_alignment', y='response_times_sess')
p1.grid(True)
# plt.scatter(times_to_head_wall_alignment, response_times_sess, s=3)
plt.plot(x, f(x,m,c), c='r', linewidth=4)
# plt.text(2,2, f"R sq = {r_sq:.2f}", color='r', fontsize=12)
p1.set_xlabel('Time to orient head to stimulus wall (s)', fontsize=15)
p1.set_ylabel('Trial response time (s)', fontsize=15)
plt.xlim([0,3.5])
plt.ylim([0,6])
plt.subplots_adjust(bottom=0.15)

plt.show()

