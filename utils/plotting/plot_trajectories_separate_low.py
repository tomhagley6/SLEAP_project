import sleap
from trajectory_extraction import extract_session_trajectories, extract_trajectory
import utils.manipulate_data.data_filtering as data_filtering
import find_frames
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import seaborn as sns
import os
import numpy as np
from utils.unused.ordinal_index import ordinal_index
from head_angle import extract_head_angle_trial
from utils.h5_file_extraction import get_locations, get_node_names
from utils.manipulate_data import data_filtering



def plot_all_trajectories_separate_low(octpy_metadata, video_path, session_trajectories, node, vectors=False, title='', grating_high=True):
    """ plot all of the trajectories in a session """

    # find which trajectories ended with choosing low
    octpy_metadata_low = data_filtering.chose_low_only(octpy_metadata)

    # make sure node is recognised
    node = node.lower()

    # for plotting the outline
    hyp = 0.82              # length of octagon wall
    hyp_div2 = hyp/2        # centre walls about x = 0
    side = 0.58             # x and y difference for diagonal octagon walls
    

    fig, ax = plt.subplots(figsize=(6,6))
    # plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    # for all trajectories given, normalise the colour map to the length of the trajectory and scatter plot the trajectory for
    # the chosen node
    for i in range(session_trajectories.shape[0]):
        timestamps = np.arange(session_trajectories[f"{node}_x"].iloc[i].shape[0])
        min_val, max_val = min(timestamps), max(timestamps)

        # if this trial is not choose low trial
        if not octpy_metadata.iloc[i].trial_num in octpy_metadata_low.trial_num.values:
            cmap = mpl.cm.summer
            norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
            sc = plt.scatter(session_trajectories[f"{node}_x"].iloc[i], session_trajectories[f"{node}_y"].iloc[i], 
                                s=2, c=timestamps, cmap=cmap, norm=norm)
        else:
            cmap = mpl.cm.copper
            norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
            sc = plt.scatter(session_trajectories[f"{node}_x"].iloc[i], session_trajectories[f"{node}_y"].iloc[i], 
                                s=2, c=timestamps, cmap=cmap, norm=norm)
            # sc = plt.scatter(session_trajectories[f"{node}_x"].iloc[i], session_trajectories[f"{node}_y"].iloc[i], 
            #                     s=2, color='grey')
        # if vectors given, plot them alongside trajectories (e.g. vector between head and neck)
        if vectors:
            plt.scatter([vectors[i][2]], [vectors[i][3]], c='b', s=3)
            plt.plot([vectors[i][0], vectors[i][2]], [vectors[i][1], vectors[i][3]], 'b-')
            plt.scatter([vectors[i][0]], [vectors[i][1]], c='r', s=7)



    # plot octagon walls
    xCoords = [hyp_div2, hyp_div2+side, hyp_div2+side, hyp_div2, -hyp_div2, -(hyp_div2+side), -(hyp_div2+side), -hyp_div2, hyp_div2]
    yCoords = [hyp_div2+side, hyp_div2, -hyp_div2, -(hyp_div2+side), -(hyp_div2+side), -hyp_div2, hyp_div2, hyp_div2+side, hyp_div2+side]
    plt.plot(xCoords[2:4], yCoords[2:4], linewidth=2)
    plt.plot(xCoords[3:5], yCoords[3:5], color='red', linewidth=2)

    # colorbar
    min_val = 0
    max_val = session_trajectories[f"{node}_x"].iloc[-1].shape[0] - 1
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(sc, cax=cax, ticks=[min_val, max_val])
    cb.ax.set_yticklabels(['start', 'end'])
    
    # plot dimensions
    plt.sca(ax)
    plt.axis('scaled')
    plt.xlim([-1.15, 1.15])
    plt.ylim([-1.15, 1.15])
    plt.gca().invert_yaxis()
    
    # plot text
    # plt.title(title)
    plt.text(-0.08, -1.03, 'high', color='red')
    plt.text(0.69, (-1.04 + 0.58/2), 'low', color='blue')
    plt.show()

    # plt.colorbar(sc)
    # plt.axis('scaled')
    # plt.xlim([-1.15, 1.15])
    # plt.ylim([-1.15, 1.15])
    # plt.gca().invert_yaxis()
    # plt.title(title)
    # plt.text(-0.08, -1.03, 'high', color='red')
    # plt.text(0.69, (-1.04 + 0.58/2), 'low', color='blue')
    # plt.show()

    return None
    