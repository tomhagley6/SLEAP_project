import sleap
from trajectory_extraction import extract_session_trajectories, extract_trajectory
import utils.manipulate_data.data_filtering as data_filtering
import find_frames
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import numpy as np
from utils.unused.ordinal_index import ordinal_index
from head_angle import extract_head_angle_trial
from utils.h5_file_extraction import get_locations, get_node_names

# TODO plotting function to plot specific subset of trajectories

def plot_all_trajectories(video_path, session_trajectories, node, vectors=False, title=''):
    """ plot all of the trajectories in a session """
    video = sleap.load_video(video_path)
    img = video[0]
    img2 = img.reshape(1080, 1440)
    cmap = mpl.cm.summer


    fig, ax = plt.subplots()
    # plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    for i in range(session_trajectories.shape[0]):
        timestamps = np.arange(session_trajectories[f"{node}_x"].iloc[i].shape[0])
        min_val, max_val = min(timestamps), max(timestamps)
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        sc = plt.scatter(session_trajectories[f"{node}_x"].iloc[i], session_trajectories[f"{node}_y"].iloc[i], s=3, c=timestamps, cmap=cmap, norm=norm)

        # if vectors given, plot them alongside trajectories (e.g. vector between head and neck)
        if vectors:
            plt.scatter([vectors[i][2]], [vectors[i][3]], c='b', s=3)
            plt.plot([vectors[i][0], vectors[i][2]], [vectors[i][1], vectors[i][3]], 'b-')
            plt.scatter([vectors[i][0]], [vectors[i][1]], c='r', s=7)


    plt.colorbar(sc)
    plt.axis('scaled')
    plt.title(title)
    plt.show()

    return None
    
if __name__ == '__main__':
    """ plot all trajectories """

    
    # params
    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    session = '2022-11-02_A006'
    trial = 0

    # plotting params
    sns.set_theme('notebook', 'ticks', font_scale=1.2)
    mpl.rcParams['figure.figsize'] = [15,6]

    directory_sleap = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
    fileName_sleap = 'model5_predictions_230206_CLI._CameraTop_2022-11-02T14-00-00.analysis.h5'
    filePath_sleap = directory_sleap + os.sep + fileName_sleap
    
    # labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-04_A004_full_model9_predictions_CLI.slp'
    labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02_A006_full_predictions_model5_CLI.slp'
    labels_file = sleap.load_file(labelsPath)

    trajectory_file_path = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports/CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
    
    # load in session data
    octpy_metadata, video_metadata_list, color_video_metadata_list \
        = find_frames.access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)

    octpy_metadata_choice = data_filtering.choice_trials_only(octpy_metadata=octpy_metadata)
    # octpy_metadata_choice = octpy_metadata
    grey_trials, grey_stims, grey_ends, \
    color_trials, color_stims, color_ends  \
        = find_frames.relevant_session_frames(octpy_metadata, video_metadata_list, color_video_metadata_list)

    trial_type = octpy_metadata.iloc[trial].trial_type
    walls = octpy_metadata.iloc[trial].wall
    locations = get_locations(trajectory_file_path)
    node_names = get_node_names(trajectory_file_path)
    trajectories, col_names = extract_trajectory(locations, node_names, grey_stims, grey_ends, trial, trial_type, walls, 0)

    session_trajectories = extract_session_trajectories(trajectory_file_path, grey_stims, grey_ends, octpy_metadata)

    video_path = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraTop_2022-11-02T14-00-00.avi'
    plot_all_trajectories(video_path, session_trajectories, 'BodyUpper')

    # trajectories for chose_low choice trials
    # track ordinal index of trials
    ordinal_index = ordinal_index(octpy_metadata)
    octpy_metadata['ordinal_index'] = ordinal_index

    # filter data and find relevant frames
    octpy_metadata_choice = data_filtering.choice_trials_only(octpy_metadata=octpy_metadata)
    octpy_metadata_choice_chose_low = data_filtering.chose_low_only(octpy_metadata=octpy_metadata_choice)
    _, grey_stims_choice_chose_low, grey_ends_choice_chose_low,_,_,_ \
          = find_frames.relevant_session_frames(octpy_metadata_choice_chose_low, video_metadata_list, color_video_metadata_list)
    
    # extract trajectories from SLEAP data  
    session_trajectories_choice_chose_low = extract_session_trajectories(trajectory_file_path, grey_stims_choice_chose_low, grey_ends_choice_chose_low, octpy_metadata)
    
    # extract head angle and head vectors for relevant trials 
    filtered_head_angles = []
    filtered_vectors = []
    for trial in range(session_trajectories_choice_chose_low.shape[0]):
        head_angles, vectors = extract_head_angle_trial(trial, grey_stims_choice_chose_low, labels_file, stim_frames_color=None, plotFlag=False, color_video_path=None)
        filtered_head_angles.append(head_angles[0])
        filtered_vectors.append(vectors[0])

    # plot
    plot_all_trajectories(video_path, session_trajectories_choice_chose_low, 'Neck', filtered_vectors)
    

    
    