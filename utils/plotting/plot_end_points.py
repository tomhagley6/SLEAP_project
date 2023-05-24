
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

def plot_end_points(session_trajectories, node, title=''):

    # make sure node is recognised
    node = node.lower()

    # for plotting the outline
    hyp = 0.82              # length of octagon wall
    hyp_div2 = hyp/2        # centre walls about x = 0
    side = 0.58             # x and y difference for diagonal octagon walls
    

    cmap = mpl.cm.summer
    x = []
    y = []

    # append the ending coordinate of each trial to a list, then plot these
    for i in range(session_trajectories.shape[0]):
        x.append(session_trajectories[f"{node}_x"].iloc[i][-1])
        y.append(session_trajectories[f"{node}_y"].iloc[i][-1])
    
    plt.figure(figsize=(6,6))
    xCoords = [hyp_div2, hyp_div2+side, hyp_div2+side, hyp_div2, -hyp_div2, -(hyp_div2+side), -(hyp_div2+side), -hyp_div2, hyp_div2]
    yCoords = [hyp_div2+side, hyp_div2, -hyp_div2, -(hyp_div2+side), -(hyp_div2+side), -hyp_div2, hyp_div2, hyp_div2+side, hyp_div2+side]
    plt.plot(xCoords[2:4], yCoords[2:4], linewidth=5)
    plt.plot(xCoords[3:5], yCoords[3:5], color='red', linewidth=5)
    plt.scatter(x, y, s=10, cmap=cmap, alpha=0.4)
    plt.axis('scaled')
    # plt.title(title)
    plt.xlim([-1.15, 1.15])
    plt.ylim([-1.15, 1.15])
    plt.gca().invert_yaxis()
    plt.text(-0.08, -1.03, 'high', color='red')
    plt.text(0.69, (-1.04 + 0.58/2), 'low', color='blue')
    plt.show()

    return None