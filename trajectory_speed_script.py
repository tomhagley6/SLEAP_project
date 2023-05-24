import find_frames
import trajectory_extraction
import numpy as np
import scipy
import os
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from utils.trajectory_speeds import trajectory_speeds, crop_speeds_session
from utils.speed_summary import find_speed_mean_STD, find_cropped_indexes, summary_stats_all_trials
from utils.manipulate_data.data_filtering import sub_15_RT_only

# globals
GREYSCALE_FRAMERATE = 60
NODE_X = "BodyUpper_x"
NODE_Y = "BodyUpper_y"

# params
trial = 10
node_x = NODE_X
node_y = NODE_Y

# paths
# for frame number extraction
data_root = '/home/tomhagley/Documents/SLEAPProject/data'
session = '2022-11-04_A004'

# for trajectory extraction
directory_trajectories = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
file_name_trajectories = 'CameraTop_2022-11-04_A004_full_model9_predictions_CLI.analysis.h5'
# file_name_trajectories = 'CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
filepath_trajectories = directory_trajectories + os.sep + file_name_trajectories

# extract frames
octpy_metadata, video_metadata, video_metadata_color = find_frames.access_metadata(data_root, session, colorVideo=True)

# filter RT
octpy_metadata = sub_15_RT_only(octpy_metadata)
grey_trials, grey_stims, grey_ends, _, _, _ \
    = find_frames.relevant_session_frames(octpy_metadata, video_metadata, video_metadata_color)

# define tracks
tracks = np.zeros(grey_stims.shape[0]).astype('int')
# extract normalised and flipped/rotated trajectories
session_trajectories = trajectory_extraction.extract_session_trajectories(filepath_trajectories, \
                                                                        grey_stims, grey_ends, octpy_metadata, tracks, \
                                                                            normalise=True, flip_rotate=True, smooth=True)

# get series of timestamps for time diff between trajectory points
timestamps_list = find_frames.timestamps_within_trial(grey_stims, grey_ends, video_metadata)

# extract and filter speed from trajectory using utils function
filtered_speeds = trajectory_speeds(session_trajectories, timestamps_list)

# crop filtered speeds to remove the first and last 5 seconds where possible
cropped_speeds = crop_speeds_session(filtered_speeds)

# find summary stats for each trial
speed_mean = []
speed_std = []
for trial in range(len(filtered_speeds)):
    speed_mean.append(find_speed_mean_STD(filtered_speeds[trial])[0])
    speed_std.append(find_speed_mean_STD(filtered_speeds[trial])[1])

#find summary stats for full session using utils function
all_trial_speeds_concat, all_trial_speed_mean, all_trial_speed_std = summary_stats_all_trials(filtered_speeds)


plt.plot(filtered_speeds[0])
plt.show()

for i in range(5):
    plt.plot(filtered_speeds[i])
    cropped_x_start, cropped_x_end = find_cropped_indexes(filtered_speeds[i])
    # if not cropped_x_start == np.nan:
    #     plt.hlines(all_trial_speed_mean, cropped_x_start, cropped_x_end, colors='r', linestyles='dashed')
    #     plt.hlines(all_trial_speed_mean + all_trial_speed_std, cropped_x_start, cropped_x_end, colors='orange', linestyles='dashed')
    #     plt.hlines(all_trial_speed_mean - all_trial_speed_std, cropped_x_start, cropped_x_end, colors='orange', linestyles='dashed')

    plt.xlabel('Time in trial (s)')
    plt.ylabel('Speed (normalised distance / s)')
    plt.show()

# plt.plot(all_trial_speeds_concat)
# plt.hlines(all_trial_speed_mean, 0, all_trial_speeds_concat.shape[0], colors='r', linestyles='dashed')
# plt.hlines(all_trial_speed_mean + 1.25*all_trial_speed_std, 0, all_trial_speeds_concat.shape[0], colors='orange', linestyles='dashed')
# plt.hlines(all_trial_speed_mean - 1.25*all_trial_speed_std, 0, all_trial_speeds_concat.shape[0], colors='orange', linestyles='dashed')

# plt.show()
