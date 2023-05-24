import numpy as np
from head_angle import angle_between_vectors
import matplotlib.pyplot as plt
from utils.distance_to_wall import distance_to_wall
from utils.get_values.get_wall_coords import get_wall_coords
from utils.get_values.get_trial_walls import get_trial_walls
from utils.manipulate_data.normalising import normalise
import math
import seaborn as sns
import pandas as pd


def trial_start_test_plot_head_ang_to_wall(octpy_metadata, trial, stim_frames, labels_file, track_num, session, plotFlag=False):
    """Given a trial and list of stim start frame numbers,
       find the angle of the head relative
       to rightward horizontal (0-degrees at Wall1) from SLEAP
       tracking data.

       Also find the coordinates for neck and nose at that frame

       And find the distance from upperbody to centre of wall_1 and 
       centre of wall_2

       Returns angle, vector
        - angle: [angle1, angle2, ... anglen]
        - vector [[nose-x,nose-y,neck-x,neck-y] ... [nose-xn,nose-yn,neck-xn,neck-yn]]
       
       If plotFlag, plot the tracking data used and the head angle
       extracted. Also plot the distance to each wall, and visualise
       the high and low wall
       
       Head vector and reference vectors have directions (head is from
       neck to nose, and reference is from left to right), assigned by 
       subtracting the 'origin' point from the 'destination' point.
       This function finds the angle between these two vectors, taking
       direction into account. 
       To preserve counterclockwise direction from Wall1, angles from 
       vectors directed below the horizontal are subtracted from 2pi."""
    
    # params
    # for plotting the outline
    hyp = 0.82              # length of octagon wall
    hyp_div2 = hyp/2        # centre walls about x = 0
    side = 0.58             # x and y difference for diagonal octagon walls

    # get index of stim start frame for chosen trial
    frame_idx = stim_frames[trial]

    # extract frame
    labels = labels_file
    labeledFrames = labels.labeled_frames
    labeledFrame = labeledFrames[frame_idx]

    track = labeledFrame.instances[track_num]

    # extract points
    points = track.points
    nose, neck = points[0], points[3]
    body = points[4]
    noseCoords = (nose.x, nose.y, nose.visible)
    neckCoords = (neck.x, neck.y, neck.visible)
        
    # for outputting
    combined_neck_nose_coords = np.array((nose.x, nose.y, neck.x, neck.y))

    # account for either point not being present
    if not nose.visible or not neck.visible:
        print("One or more points are invisible in frame.")
        return None


    #vector
    # head direction vector
    neckNoseVector = np.array((nose.x - neck.x, nose.y - neck.y))

    # horizontal vector
    centre = np.array((699, 532))
    rightMiddle = np.array((1440, 532))
    rightVector = rightMiddle - centre

    # top-left used for normalisation
    top_left = np.array((227, 66))



    result = angle_between_vectors(np.hstack([neckNoseVector, rightVector]))

    # angle between vectors is symmetric above and below the horizontal vector (formula gives minimum angle)
    # so subtract angle from 2pi if below horizontal vector
    yPositive = neckNoseVector[1] > 0
    if yPositive:     # yPositive when nose is closer to bottom of image than neck is 
        result = (np.pi*2) - result

    # new code - find distances 
    trial_type = octpy_metadata.iloc[trial].trial_type
    walls = octpy_metadata.iloc[trial].wall
    wall_x_coords, wall_y_coords = get_wall_coords()
    trial_walls = get_trial_walls(trial_type, walls)
    wall_1_coords = [wall_x_coords[trial_walls[0] - 1], wall_y_coords[trial_walls[0] - 1]]
    wall_2_coords = [wall_x_coords[trial_walls[1] - 1], wall_y_coords[trial_walls[1] - 1]]
    # wall_1_norm = normalise(wall_1_coords[0], wall_1_coords[1])
    # wall_2_norm = normalise(wall_2_coords[0], wall_2_coords[1])
    node_norm = normalise(body.x, body.y)

    distance_wall_1 = distance_to_wall(node_norm[0], node_norm[1], wall_1_coords[0], wall_1_coords[1])
    distance_wall_2 = distance_to_wall(node_norm[0], node_norm[1], wall_2_coords[0], wall_2_coords[1])

    if plotFlag:
 



        #fig, ax = plt.subplot(2,1)
        img = labeledFrame.image

        fig0, ax0 = plt.subplots()
        ax0.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')
        # ax0.scatter(wall_x_coords, wall_y_coords, s=6, c='y')
        plt.scatter([centre[0], rightMiddle[0], top_left[0]], [centre[1], rightMiddle[1], top_left[1]], s=6, c='r')

        # sns.scatterplot(pd.DataFrame({"x Coordinate":wall_x_coords, "y Coordinate":wall_y_coords}),
        #                  x="x Coordinate", y="y Coordinate", s=6, c='y')

        ax0.imshow(img, cmap='gray')
        ax0.set_xlabel('x Coordinate')
        ax0.set_ylabel('y Coordinate')
        ax0.set_frame_on(False)
        ax0.axis('off')
        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)

        sns.set(font_scale=1.2)
        sns.set_style("dark")

        plt.show()


        sns.set(font_scale=1.2)
        sns.set_style("dark")
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(221)
        plt.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')
        # sns.scatterplot(pd.DataFrame({"x Coordinate":[nose.x, neck.x], "y Coordinate":[nose.y, neck.y]}), x="x Coordinate", y="y Coordinate", s=3)
        # plt.scatter([centre[0], rightMiddle[0], top_left[0]], [centre[1], rightMiddle[1], top_left[1]], s=3, c='r')
        # plt.scatter(wall_x_coords, wall_y_coords, s=3, c='y')
        ax1.imshow(img, cmap='gray')

        ax1.set_xlabel('x Coordinate')
        ax1.set_ylabel('y Coordinate')
        ax1.set_frame_on(False)
        ax1.set_title('Coordinate locations')
        plt.subplots_adjust(bottom=0.04)
        plt.subplots_adjust(top=0.90)


        # specify polar axes by polar=True or projection='polar'
        ax2 = fig.add_subplot(222, projection='polar')

        theta = result
        bar = ax2.bar(theta, 1, width=np.pi/16, bottom=0.0, alpha=0.5)
        #bar.set_alpha(0.5)
        ax2.set_yticklabels([])
        ax2.set_title('Head angle')

        sns.set_style("darkgrid")
        ax3 = fig.add_subplot(223)
        ax3.bar(['Distance to high', 'Distance to low'],[session.distances_wall_1[trial],
                                                         session.distances_wall_2[trial]], width=0.5, color='orange')
        ax3.set_ylim([0,2])
        ax3.set_ylabel("Distance")
        ax3.set_frame_on(True)
        ax3.set_title('Distance to trial walls')
        ax3.axes.xaxis.grid(False)

        ax4 = fig.add_subplot(224)
        ax4.bar(['Head angle to high', 'Head angle to low'],[np.degrees(session.head_angles_wall_1[trial]), 
                                                             np.degrees(session.head_angles_wall_2[trial])], width=0.5, color='purple')
        ax4.set_ylim([0,180])
        ax4.set_ylabel("Angle (°)")
        ax4.set_frame_on(True)
        ax4.set_title('Head angle to trial walls')
        ax4.axes.xaxis.grid(False)


        # # Commented out to make presentation figures
        # # plot octagon walls
        # # 
        # ax4 = fig.add_subplot(224)
        # xCoords = [hyp_div2, hyp_div2+side, hyp_div2+side, hyp_div2, -hyp_div2, -(hyp_div2+side), -(hyp_div2+side), -hyp_div2, hyp_div2]
        # yCoords = [hyp_div2+side, hyp_div2, -hyp_div2, -(hyp_div2+side), -(hyp_div2+side), -hyp_div2, hyp_div2, hyp_div2+side, hyp_div2+side]
        # # wall 6 is 0:2, wall 7 is 1:3, etc.

        # # plot wall 1
        # # find starting index
        # # list element num is wall num + 1, so 8 should be reset to 0
        # idx = trial_walls[0]
        # if idx == 8:
        #     idx = 0
        # ax4.plot(xCoords[idx:idx+2], yCoords[idx:idx+2], color='red', linewidth=2)

        # # plot wall 2
        # # find starting index
        # # list element num is wall num + 1, so 8 should be reset to 0
        # idx = trial_walls[1]
        # if idx == 8:
        #     idx = 0
        # ax4.plot(xCoords[idx:idx+2], yCoords[idx:idx+2], color='blue', linewidth=2)
        # plt.sca(ax4)
        # plt.axis('scaled')
        # plt.xlim([-1.15, 1.15])
        # plt.ylim([-1.15, 1.15])
        # ax4.set_frame_on(True)
        # ax4.set_xticks([])
        # ax4.set_yticks([])
        # ax4.set_title('Trial high and low walls')
        # #plt.gca().invert_yaxis()


        plt.show()


    angle = result
    vector = combined_neck_nose_coords

    return angle, vector
