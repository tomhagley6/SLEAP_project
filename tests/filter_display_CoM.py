from utils.manipulate_data.data_filtering import select_trials_only, choice_trials_only
import numpy as np
import os
import sleap
import find_frames
import trajectory_extraction
import analysis.trajectory_analysis
from utils.plotting.plot_all_trajectories import plot_all_trajectories

# PARAMS
SOCIAL = False
if SOCIAL:
    project_type = 'octagon_multi'
else:
    project_type = 'octagon_solo'

session = '2022-11-04_A004'

# PATHS
# trial metadata
data_root = '/home/tomhagley/Documents/SLEAPProject/data'

# greyscale video
video_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/videos/{session}'
video_filename = f'CameraTop_{session}' + '_all.avi'
video_path_greyscale = video_root + os.sep + video_filename

# tracking data
trajectories_root = f'/home/tomhagley/Documents/SLEAPProject/{project_type}/exports/{session}'
trajectories_filename = f'CameraTop_{session}' + '_analysis.h5'
trajectories_filepath = trajectories_root + os.sep + trajectories_filename

trajectories_filepath = trajectories_root + os.sep + trajectories_filename

# load metadata files from HPC using octpy
octpy_metadata, video_metadata, color_video_metadata = find_frames.access_metadata(data_root, session, 
                                                                                   colorVideo=True, refreshFiles=False)

# filter data if required (e.g. RT < 15 s, choice-trials only)
octpy_metadata = choice_trials_only(octpy_metadata)
# CoM_trial_idxs = np.array([ 18, 25, 134, 155, 164])
CoM_trial_idxs = np.array([ 155])
om_select = select_trials_only(octpy_metadata, CoM_trial_idxs)


# find frames for trials
_, grey_stims, grey_ends, _, _, _ = find_frames.relevant_session_frames(om_select, video_metadata, color_video_metadata)

# set tracks
tracks = np.zeros(om_select.shape[0]).astype('int')

trajectories = trajectory_extraction.extract_session_trajectories(trajectories_filepath, grey_stims, grey_ends,
                                                   om_select, tracks, normalise=True, flip_rotate=True, smooth=True)

title = 'Change of mind trials trajectories for a single session'
plot_all_trajectories(video_path_greyscale, trajectories, 'BodyUpper', vectors=False, title=title)