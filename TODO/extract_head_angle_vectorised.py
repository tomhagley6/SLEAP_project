import sleap
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
from find_frames import access_metadata, relevant_session_frames
from head_angle import angle_between_vectors


# not so important - can just take labels file loading out of the function

def extract_head_angle_vectorised(data_root, session, labelsPath, plotFlag=False):
    """Given a frame index, find the angle of the head relative
       to rightward horizontal (0-degrees at Wall1) from SLEAP
       tracking data.
       
       If plotFlag, plot the tracking data used and the head angle
       extracted.
       
       Head vector and reference vectors have directions (head is from
       neck to nose, and reference is from left to right), assigned by 
       subtracting the 'origin' point from the 'destination' point.
       This function finds the angle between these two vectors, taking
       direction into account. 
       To preserve counterclockwise direction from Wall1, angles from 
       vectors directed below the horizontal are subtracted from 2pi."""
    
    octpyMetadata, videoMetadata_list, colorVideoMetadata_list = access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)
    _, stim_frames, _ = relevant_session_frames(octpyMetadata, videoMetadata_list, colorVideoMetadata_list, 10)

    # extract frame
    labels = sleap.load_file(labelsPath)
    all_labeled_frames = labels.labeled_frames
    labeled_frames = all_labeled_frames[stim_frames.values]

    # identify social/solo
    social = True
    tracks = labels.tracks
    if len(tracks) == 1:
        social = False

    # current frame track instances
    if social:
        try:
            track0, track1 = labeled_frames.instances
        except IndexError:
            print("Less than expected number of instances in frame")
    else:
        try:
            track0 = labeled_frames.instances[0]
        except IndexError:
            print("Less than expected number of instances in frame")

    # extract points
    points = track0.points
    nose, neck = points[0], points[3]
    noseCoords = (nose.x, nose.y, nose.visible)
    neckCoords = (neck.x, neck.y, neck.visible)

    # account for either point not being present
    if not nose.visible or not neck.visible:
        print("One or more points are invisible in frame.")
        return None


    #vectors
    # head direction vector
    neckNoseVector = np.array((nose.x - neck.x, nose.y - neck.y))
    # neckNoseVector = np.array((neck.x - nose.x, neck.y - nose.y))

    # horizontal vector
    centre= np.array((699, 532))
    rightMiddle = np.array((1440, 532))
    rightVector = rightMiddle - centre
    
    # # replaced with horizontal
    # up vector
    # topMiddle= np.array((650,1080))
    # upVector = topMiddle - centre 

    result = angle_between_vectors(np.hstack([neckNoseVector, rightVector]))
    
    # angle between vectors is symmetric above and below the horizontal vector (formula gives minimum angle)
    # so subtract angle from 2pi if below horizontal vector
    yPositive = neckNoseVector[1] > 0
    if yPositive:     # yPositive when nose is closer to bottom of image than neck is 
        result = (np.pi*2) - result

    if plotFlag:
        # plot
        # img = labeled_frames.image
        # plt.figure(1)
        # plt.imshow(img, cmap='gray')
        # plt.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')
        # plt.show()

        #fig, ax = plt.subplot(2,1)
        img = labeled_frames.image
        fig = plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(211)
        ax1.imshow(img, cmap='gray')
        plt.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')

        # specify polar axes by polar=True or projection='polar'
        ax2 = fig.add_subplot(212, projection='polar')
        #ax = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
        #ax2.set_theta_zero_location('N')

        theta = result
        bar = ax2.bar(theta, 1, width=np.pi/16, bottom=0.0, alpha=0.5)
        #bar.set_alpha(0.5)
        plt.show()

    print("frame completed")
    return result


if __name__ == '__main__':
    labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02T14-00-00.avi_model5_predictions_230206_CLI.slp'
    frameIdx = 7383
    trial = 10

    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    session = '2022-11-02_A006'

    result = extract_head_angle_vectorised(data_root=data_root, session=session, labelsPath=labelsPath)