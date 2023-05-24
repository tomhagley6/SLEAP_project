from utils.change_of_mind import change_of_mind_session
import numpy as np

""" functions to filter octpy metadata """


def choice_trials_only(octpy_metadata):
    om_choice = octpy_metadata[octpy_metadata['choice_trial'] == True]
    
    return om_choice

def sub_x_RT_only(octpy_metadata, x):
    """ filter trials with response time > x seconds """
    om_sub_x_RT = octpy_metadata[octpy_metadata['RT'] < x]

    return om_sub_x_RT

def chose_light_only(octpy_metadata):
    om_chose_light = octpy_metadata[octpy_metadata['chose_light'] == True]
    
    return om_chose_light

def filter_miss_trials(octpy_metadata):
    om_hit_trials_only = octpy_metadata[octpy_metadata['miss_trial'] == False] 

    return om_hit_trials_only

def select_trials_only(octpy_metadata, idx_list):
    om_selection = octpy_metadata.iloc[idx_list]

    return om_selection

def crop_ends(octpy_metadata):
    """ crop 2 trials from either end """
    octpy_metadata_cropped = octpy_metadata[2:-2]

    return octpy_metadata_cropped

def chose_low_only(octpy_metadata, gratingHigh = True):
    if gratingHigh:
        om_chose_low = octpy_metadata[octpy_metadata['chose_light'] == True]
    else:
        om_chose_low = octpy_metadata[octpy_metadata['chose_light'] == False]
        
    return om_chose_low

def change_of_mind_trials(session_trajectories, video_trajectories, labels_file, octpy_metadata, 
                    stim_frames, end_frames, tracks, timestamps_list, video_root, video_filename,
                    color_frame_offset, save_videos=False):


    CoM_trial_idxs = change_of_mind_session(session_trajectories, video_trajectories, labels_file, octpy_metadata, 
                    stim_frames, end_frames, tracks, timestamps_list, video_root, video_filename,
                    color_frame_offset, save_videos=save_videos)
   
    CoM_trials_mask = np.zeros(octpy_metadata.shape[0])
    for trial in range(octpy_metadata.shape[0]):
       if trial in CoM_trial_idxs:
           CoM_trials_mask[trial] = 1

    octpy_metadata['CoM_trial'] = CoM_trials_mask
    octpy_metadata_CoM = octpy_metadata[octpy_metadata['CoM_trial'] == 1]
   
    return octpy_metadata_CoM

def non_change_of_mind_trials(session_trajectories, video_trajectories, labels_file, octpy_metadata, 
                    stim_frames, end_frames, tracks, timestamps_list, video_root, video_filename,
                    color_frame_offset, save_videos=False):


    CoM_trial_idxs = change_of_mind_session(session_trajectories, video_trajectories, labels_file, octpy_metadata, 
                    stim_frames, end_frames, tracks, timestamps_list, video_root, video_filename,
                    color_frame_offset, save_videos=save_videos)
   
    CoM_trials_mask = np.zeros(octpy_metadata.shape[0])
    for trial in range(octpy_metadata.shape[0]):
       if trial in CoM_trial_idxs:
           CoM_trials_mask[trial] = 1

    octpy_metadata['CoM_trial'] = CoM_trials_mask
    octpy_metadata_non_CoM = octpy_metadata[octpy_metadata['CoM_trial'] == 0]
   
    return octpy_metadata_non_CoM