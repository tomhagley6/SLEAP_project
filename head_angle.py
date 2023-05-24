import sleap
import numpy as np
import math
import matplotlib.pyplot as plt
from find_frames import access_metadata, relevant_session_frames
import utils.manipulate_data.data_filtering as data_filtering
import cv2
import warnings


""" Functions to find head angle of trials, wall angles for all walls, 
    the difference between head angle and wall angles, and summary statistics
    on this """

def extract_head_angle_session(stim_frames, labels_file, tracks, plotFlag=False, social=False):
    
    """ find head angle for each trial in session, 
        defined as a Series of stimulus start frame numbers
         
        Wrapper function for extract_head_angle_trial """
    
    if social: # TODO
        pass
    else:
        head_angles = []
        for trial_idx in range(len(stim_frames)):
            # take the first (and only) entry in the list of head angles, the first return of extract_head_angle 
            head_angles.append(extract_head_angle_trial(trial_idx, stim_frames, labels_file=labels_file, track_num=tracks[trial_idx], plotFlag=plotFlag)[0])
        
        return head_angles

def extract_head_vector_session(stim_frames, labels_file, tracks, plotFlag=False, social=False):
    
    """ find head vector for each trial in session, 
        defined as a Series of stimulus start frame numbers
         
        Wrapper function for extract_head_angle_trial """
    if social: # TODO
        pass
    else:
        head_vectors = []
        for trial_idx in range(len(stim_frames)):
            # take vector output, not head angle
            head_vectors.append(extract_head_angle_trial(trial_idx, stim_frames, labels_file=labels_file, track_num=tracks[trial_idx], plotFlag=plotFlag)[1])
        
        return head_vectors


def extract_head_angle_trial(trial, stim_frames, labels_file, track_num, stim_frames_color=None, plotFlag=False, color_video_path=None):
    """Given a trial and list of stim start frame numbers,
       find the angle of the head relative
       to rightward horizontal (0-degrees at Wall1) from SLEAP
       tracking data.

       Also find the coordinates for neck and nose at that frame

       Returns angle, vector
        - angle: [angle1, angle2, ... anglen]
        - vector [[nose-x,nose-y,neck-x,neck-y] ... [nose-xn,nose-yn,neck-xn,neck-yn]]
       
       If plotFlag, plot the tracking data used and the head angle
       extracted.
       
       Head vector and reference vectors have directions (head is from
       neck to nose, and reference is from left to right), assigned by 
       subtracting the 'origin' point from the 'destination' point.
       This function finds the angle between these two vectors, taking
       direction into account. 
       To preserve counterclockwise direction from Wall1, angles from 
       vectors directed below the horizontal are subtracted from 2pi."""

    # get index of stim start frame for chosen trial
    frame_idx = stim_frames[trial]

    # extract frame
    labels = labels_file
    labeledFrames = labels.labeled_frames
    labeledFrame = labeledFrames[frame_idx]

    # account for no instance in this frame
    while len(labeledFrame.instances) == 0:
        warnings.warn(f"No instance found in frame {frame_idx} for trial {trial}. Skipping forward by 1 until instance found.")
        try:
            labeledFrame = labeledFrames[frame_idx + 1]
        except:
            warnings.warn(f"Could not extract head angle for frame {frame_idx} in trial {trial}.")
            angle = np.nan
            vector = np.array([np.nan, np.nan])
            
            return angle, vector

    track = labeledFrame.instances[track_num]

    # extract points
    points = track.points
    nose, neck = points[0], points[3]
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
        # img = labeledFrame.image
        # plt.figure(1)
        # plt.imshow(img, cmap='gray')
        # plt.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')
        # plt.show()

        #fig, ax = plt.subplot(2,1)
        img = labeledFrame.image
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

        # if stim_frames_color:
        #     fig, ax = plt.subplots()
        #     cap = cv2.VideoCapture

        #     ax.plot

    angle = result
    vector = combined_neck_nose_coords

    return angle, vector

# UNUSED
def extract_head_angle_trial_solo_OLD(trial, stim_frames, labels_file, stim_frames_color=None, plotFlag=False, color_video_path=None):
    """Given a trial and list of stim start frame numbers,
       find the angle of the head relative
       to rightward horizontal (0-degrees at Wall1) from SLEAP
       tracking data.

       Also find the coordinates for neck and nose at that frame

       Returns angle, vector
        - angle: [angle1, angle2, ... anglen]
        - vector [[nose-x,nose-y,neck-x,neck-y] ... [nose-xn,nose-yn,neck-xn,neck-yn]]
       
       If plotFlag, plot the tracking data used and the head angle
       extracted.
       
       Head vector and reference vectors have directions (head is from
       neck to nose, and reference is from left to right), assigned by 
       subtracting the 'origin' point from the 'destination' point.
       This function finds the angle between these two vectors, taking
       direction into account. 
       To preserve counterclockwise direction from Wall1, angles from 
       vectors directed below the horizontal are subtracted from 2pi."""

    # get index of stim start frame for chosen trial
    frame_idx = stim_frames[trial]

    # extract frame
    labels = labels_file
    labeledFrames = labels.labeled_frames
    labeledFrame = labeledFrames[frame_idx]

    track = labeledFrame.instances[0]

    # extract points
    points = track.points
    nose, neck = points[0], points[3]
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
    # neckNoseVector = np.array((neck.x - nose.x, neck.y - nose.y))

    # horizontal vector
    centre= np.array((650,520))
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
        # img = labeledFrame.image
        # plt.figure(1)
        # plt.imshow(img, cmap='gray')
        # plt.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')
        # plt.show()

        #fig, ax = plt.subplot(2,1)
        img = labeledFrame.image
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

        # if stim_frames_color:
        #     fig, ax = plt.subplots()
        #     cap = cv2.VideoCapture

        #     ax.plot

    angle = result
    vector = combined_neck_nose_coords

    return angle, vector


# UNUSED
def extract_head_angle_trial_OLD(trial, stim_frames, labels_file, stim_frames_color=None, plotFlag=False, color_video_path=None):
    """Given a trial and list of stim start frame numbers,
       find the angle of the head relative
       to rightward horizontal (0-degrees at Wall1) from SLEAP
       tracking data.

       Also find the coordinates for neck and nose at that frame

       Returns angle, vector
        - angle: [angle1, angle2, ... anglen]
        - vector [[nose-x,nose-y,neck-x,neck-y] ... [nose-xn,nose-yn,neck-xn,neck-yn]]
       
       If plotFlag, plot the tracking data used and the head angle
       extracted.
       
       Head vector and reference vectors have directions (head is from
       neck to nose, and reference is from left to right), assigned by 
       subtracting the 'origin' point from the 'destination' point.
       This function finds the angle between these two vectors, taking
       direction into account. 
       To preserve counterclockwise direction from Wall1, angles from 
       vectors directed below the horizontal are subtracted from 2pi."""

    # get index of stim start frame for chosen trial
    frame_idx = stim_frames[trial]

    # extract frame
    labels = labels_file
    labeledFrames = labels.labeled_frames
    labeledFrame = labeledFrames[frame_idx]

    # identify social/solo
    social = True
    tracks = labels.tracks
    if len(tracks) == 1:
        social = False

    # current frame track instances
    if social:
        try:
            track0, track1 = labeledFrame.instances
            tracks = [track0, track1]
        except IndexError:
            print("Less than expected number of instances in frame")
    else:
        try:
            track0 = labeledFrame.instances[0]
            tracks = [track0]
        except IndexError:
            print("Less than expected number of instances in frame")

    vectors = []
    angles = []
    for track in range(len(tracks)):
        # extract points
        points = tracks[track].points
        nose, neck = points[0], points[3]
        noseCoords = (nose.x, nose.y, nose.visible)
        neckCoords = (neck.x, neck.y, neck.visible)
        
        # for outputting
        combined_neck_nose_coords = np.array((nose.x, nose.y, neck.x, neck.y))

        # account for either point not being present
        if not nose.visible or not neck.visible:
            print("One or more points are invisible in frame.")
            return None


        #vectors
        # head direction vector
        neckNoseVector = np.array((nose.x - neck.x, nose.y - neck.y))
        # neckNoseVector = np.array((neck.x - nose.x, neck.y - nose.y))

        # horizontal vector
        centre= np.array((650,520))
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
            # img = labeledFrame.image
            # plt.figure(1)
            # plt.imshow(img, cmap='gray')
            # plt.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')
            # plt.show()

            #fig, ax = plt.subplot(2,1)
            img = labeledFrame.image
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

            # if stim_frames_color:
            #     fig, ax = plt.subplots()
            #     cap = cv2.VideoCapture

            #     ax.plot

        angles.append(result)
        vectors.append(combined_neck_nose_coords)

    return angles, vectors

# TODO social
def extract_head_angle_frame(frame_num, labels_file, track_num, stim_frames_color=None, plotFlag=False, color_video_path=None):
    """Given a frame,
       find the angle of the head relative
       to rightward horizontal (0-degrees at Wall1) from SLEAP
       tracking data.

       Also find the coordinates for neck and nose at that frame

       Returns angle, vector
        - angle: [angle1, angle2, ... anglen]
        - vector [[nose-x,nose-y,neck-x,neck-y] ... [nose-xn,nose-yn,neck-xn,neck-yn]]
       
       If plotFlag, plot the tracking data used and the head angle
       extracted.
       
       Head vector and reference vectors have directions (head is from
       neck to nose, and reference is from left to right), assigned by 
       subtracting the 'origin' point from the 'destination' point.
       This function finds the angle between these two vectors, taking
       direction into account. 
       To preserve counterclockwise direction from Wall1, angles from 
       vectors directed below the horizontal are subtracted from 2pi."""

 # get index of stim start frame for chosen trial
    frame_idx = frame_num

    # extract frame
    labels = labels_file
    labeledFrames = labels.labeled_frames
    labeledFrame = labeledFrames[frame_idx]
    
    # account for no instance in this frame
    count = 1
    while len(labeledFrame.instances) == 0:
        warnings.warn(f"No instance found in frame {frame_idx} for video. Skipping forward by 1 until instance found.")
        try:
            labeledFrame = labeledFrames[frame_idx + count]
            count+=1
        except:
            warnings.warn(f"Could not extract head angle for frame {frame_idx} in video.")
            angle = np.nan
            vector = np.array([np.nan, np.nan])
            
            return angle, vector
    try:
        track = labeledFrame.instances[track_num]
    except IndexError:
            warnings.warn(f"IndexError in head angle extraction at {frame_idx} in video.")
            angle = np.nan
            vector = np.array([np.nan, np.nan])            
            
            return angle, vector 

    # extract points
    # careful - point indexes  change based on any missing points
    # currently assuming all points are visible in each frame
    points = track.points
    try:
        nose, neck = points[0], points[3]
    except IndexError:
        nose, neck = np.nan, np.nan
        return np.nan, np.nan
    # noseCoords = (nose.x, nose.y, nose.visible)
    # neckCoords = (neck.x, neck.y, neck.visible)
        
    # for outputting
    # combined_neck_nose_coords = np.array((nose.x, nose.y, neck.x, neck.y))

    # account for either point not being present
    if not nose.visible or not neck.visible:
        print("One or more points are invisible in frame.")
        warnings.warn(f"One or more points not visible in frame {frame_idx}.")
        return None


    #vector
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
        # img = labeledFrame.image
        # plt.figure(1)
        # plt.imshow(img, cmap='gray')
        # plt.scatter([nose.x, neck.x], [nose.y, neck.y], s=3, c='b')
        # plt.show()

        #fig, ax = plt.subplot(2,1)
        img = labeledFrame.image
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

        # if stim_frames_color:
        #     fig, ax = plt.subplots()
        #     cap = cv2.VideoCapture

        #     ax.plot

    angle = result
    vector = neckNoseVector

    return angle, vector

def extract_head_angle_frame_session(labels_file, track_num):
    """ loop through extract_head_angle_frame for all the frames in a single session """
    # find n_frames from the labels file
    n_frames = len(labels_file.labeled_frames)
    head_angle_frames = []
    for frame_num in range(n_frames):
        head_angle_frames.append(extract_head_angle_frame(frame_num, labels_file, 
                                                        track_num, stim_frames_color=None, 
                                                        plotFlag=False, color_video_path=None)[0])
        
    return np.array(head_angle_frames)

# UNUSED
def color_video_frame(video_path, stim_frames, trial, color_delay):
    
    # save movie frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, stim_frames.iloc[trial] - color_delay + 4)
    ret, image = cap.read()
    file_name = f"frame {stim_frames.iloc[trial]}.jpg"
    cv2.imwrite(file_name, image)
    print("frame saved")

    # display movie frame
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    plt.show()

    return file_name


def get_wall_angles(plot_flag=False):
    """ Return angles of the 8 walls from rightward horizontal """
    # horizontal vector
    centre= np.array((699, 532))
    rightMiddle = np.array((1440, 532))
    rightVector = rightMiddle - centre

    # starting from rightmost wall and going clockwise
    wall_ycoords = [547, 205, 63, 208, 549, 885, 1023, 884]
    wall_xcoords = [1177, 1041, 698, 360, 220, 363, 700, 1036]

    # conver from coordinats to a vector from centre
    wall_ycoords_vector = [y_coord - centre[1] for y_coord in wall_ycoords]
    wall_xcoords_vector = [x_coord - centre[0] for x_coord in wall_xcoords]
    wall_vectors = list(zip(wall_xcoords_vector, wall_ycoords_vector))

    # use the vector find angle from horizontal
    wall_angles = []
    for i in range(len(wall_ycoords)):
        difference  = angle_between_vectors(np.hstack([wall_vectors[i], rightVector]))
        wall_angles.append((difference))

    # angle between vectors is symmetric above and below the horizontal vector (formula gives minimum angle)
    # so subtract angle from 2pi if below horizontal vector
    for i in range(len(wall_angles)):
        if wall_vectors[i][1] < 0:  # if y negative
            wall_angles[i] = 2*np.pi - wall_angles[i]

    if plot_flag:
        # plot
        fig = plt.figure(figsize=(12,12))
        # specify polar axes by polar=True or projection='polar'
        ax2 = fig.add_subplot(111, projection='polar')
        labels =  [f"Wall {i}" for i in range(8)]
        bars = ax2.bar(wall_angles, [1]*(len(wall_angles)), width=np.pi/16, bottom=0.0, alpha=0.5)
        ax2.bar_label(bars, labels=labels, padding=3)
        #bar.set_alpha(0.5)
        plt.show()

    return wall_angles


# pd apply function
# Assumes grating is high!
def order_walls(walls, trial_type):
    """ reverse order of walls in OctPy if trial_type is LG """
    intWalls = [int(x) for x in walls.split(',')]
    if trial_type == 'choiceLG':
        intWalls.reverse()
    
    return intWalls

# pd apply function
def smallest_head_angle_to_wall(head_angle, wall_angle):
    """ find the obtuse angle between the head and the wall """
    difference = abs(head_angle - wall_angle)
    if difference > math.pi:
        difference = 2*math.pi - difference
    
    return difference

# Outdated and incorrect, see utils.head_angle_to_wall
def head_angle_to_wall_OLD(head_ang, wall_nums, wall_angles):
    """ Given the head angle and wall numbers, and wall angles,
        find the head angle to this wall
         
        return floats (head angle to wall 1, head angle to wall 2) """

    # index wall angles for wall_nums
    wall_ang_1 = wall_angles[wall_nums[0] - 1]
    wall_ang_2 = wall_angles[wall_nums[1] - 1]
    
    # find the obtuse angle between the head and the wall 
    difference_wall_1 = abs(head_ang - wall_ang_1)
    if difference_wall_1 > math.pi:
        difference_wall_1 = 2*math.pi - difference_wall_1

    difference_wall_2 = abs(head_ang - wall_ang_2)
    if difference_wall_2 > math.pi:
        difference_wall_2 = 2*math.pi - difference_wall_2

    return difference_wall_1, difference_wall_2


# UNUSED
def head_angle_to_wall_stim_on(octpy_metadata, stim_frames, labels_file, tracks, wall_angles, checks=False, check_trial=10):
    """ Update octpy metadata to include head angle to the walls """

    om = octpy_metadata

    # get stim start head angles for the session
    om['head_angle_stim'] = extract_head_angle_session(stim_frames, labels_file, tracks, plotFlag=False)
    
    # extract wall info from om
    # list comprehension plus map to parse pandas series data
    # for every row, map int to the iterable, returning an iterator which we extract a list from
    walls = [list(map(int, i.split(','))) for i in om['wall']] 
    walls = np.array(walls)
    om['walls_ordered'] = om.apply(lambda x: order_walls(x['wall'], x['trial_type']), axis=1)
    om['wall_high'] = om.apply(lambda x: x['walls_ordered'][0], axis=1)
    om['wall_low'] = om.apply(lambda x: x['walls_ordered'][1], axis=1)

    # identify whether chose high
    # currently assuming grating is high!
    om['chose_high'] = np.where(om['chose_light'] == False, True, False)

    # index wall angles in each trial
    om['wall_high_ang'] = om.apply(lambda x: wall_angles[x['wall_high'] - 1], axis=1)
    om['wall_low_ang'] = om.apply(lambda x: wall_angles[x['wall_low'] - 1], axis=1)

    om['head_ang_wall_high'] = om.apply(lambda x: smallest_head_angle_to_wall(x['head_angle_stim'], x['wall_high_ang']), axis=1)
    om['head_ang_wall_low'] = om.apply(lambda x: smallest_head_angle_to_wall(x['head_angle_stim'], x['wall_low_ang']), axis=1)

    om['ang_close_to_low'] = om.apply(lambda x: x['head_ang_wall_high'] > x['head_ang_wall_low'], axis=1)

    ## checks
    if checks:
        om_checks = om.iloc[check_trial]
        print(f"Wall high is: {om_checks['wall_high']} with angle {wall_angles[om_checks['wall_high'] - 1]:.3f} from horizontal")
        print(f"Above angle should be identical to {om_checks['wall_high_ang']:.3f}")
        print(f"Wall low is: {om_checks['wall_low']} with angle {wall_angles[om_checks['wall_low'] - 1]:.3f} from horizontal")
        print(f"Above angle should be identical to {om_checks['wall_low_ang']:.3f}")
        print(f"Stim start head angle is: {om_checks['head_angle_stim']:.3f}")
        print(f"Angle between the head and wall high is {om_checks['head_ang_wall_high']:.3f}")
        print(f"Angle between the head and wall low is {om_checks['head_ang_wall_low']:.3f}")
        print(f"Given these two values, 'ang_close_to_low' == {om_checks['ang_close_to_low']}")

    return om


def angle_between_vectors(coords):
    """ return angle between 2 vectors """
    a,b,c,d = coords
    v1, v2 = np.array((a,b)), np.array((c,d))

    dotProduct = v1[0]*v2[0] + v1[1]*v2[1]
    v1Magnitude_sq = v1[0]**2 + v1[1]**2
    v2Magnitude_sq = v2[0]**2 + v2[1]**2
    v1Magnitude = math.sqrt(v1Magnitude_sq)
    v2Magnitude = math.sqrt(v2Magnitude_sq)

    cosTheta = dotProduct/(v1Magnitude*v2Magnitude)
    theta = math.acos(cosTheta)
    
    return theta

if __name__ == '__main__':
    """ show all wall angles, show example head-angle extraction frame with check, color camera stim_onset frame,
        and comparison between P(high) for head angle_close_to_high vs low """


    labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.slp'
    labels_file = sleap.load_file(labelsPath)
    color_video_path = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraColorTop_2022-11-02T14-00-00.avi'
    frameIdx = 7383
    wall_angles = get_wall_angles()
    trial = 7
    color_delay = 52

    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    session = '2022-11-02_A006'
    octpyMetadata, videoMetadata_list, colorVideoMetadata_list = access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)
    # currently testing with a filtered dataset
    octpy_metadata_choice = data_filtering.choice_trials_only(octpy_metadata=octpyMetadata)
    greyTrials_choice, greyStims_choice, greyEnds_choice, colorTrials_choice, colorTrials_stim, _ = relevant_session_frames(octpy_metadata_choice, videoMetadata_list, colorVideoMetadata_list)

    # replace frame index with trial for easy checking
    angle, vector = extract_head_angle_trial(trial=trial, stim_frames=greyStims_choice, labels_file=labels_file, track_num=0, plotFlag=True)

    # show color video frame too 
    file_path = color_video_frame(color_video_path, colorTrials_stim, trial=trial, color_delay=color_delay)



    # result = angle_between_vectors((0,560, -200, -12))
    print(math.degrees(angle), angle)

    head_angles = extract_head_angle_session(stim_frames=greyStims_choice, labels_file=labels_file, track_num=0, plotFlag=False)

    # octpy_metadata for choice trials only, with useful extracted values for angles
    # om_choice = head_angle_to_wall(data_root, session, labels_file, wall_angles)

    octpy_metadata, videoMetadata_list, colorVideoMetadata_list = access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=False)
    octpy_metadata_choice = data_filtering.choice_trials_only(octpy_metadata=octpy_metadata)
    _, stim_frames_choice, _, _, _, _ = relevant_session_frames(octpy_metadata_choice, videoMetadata_list, colorVideoMetadata_list)
    om_choice = head_angle_to_wall_stim_on(octpy_metadata=octpy_metadata_choice, stim_frames=stim_frames_choice, labels_file=labels_file, \
                                   wall_angles=wall_angles, track_num=0, checks=True, check_trial=trial)

    # visualise ang_close_to_low
    # proportion of trials chose high
    chose_high_oriented_to_low = om_choice["chose_high"][om_choice["ang_close_to_low"] == True]
    chose_high_oriented_to_high = om_choice["chose_high"][om_choice["ang_close_to_low"] == False]
    proportion_chose_high_oriented_to_low = chose_high_oriented_to_low.sum()/chose_high_oriented_to_low.shape[0]
    proportion_chose_high_oriented_to_high = chose_high_oriented_to_high.sum()/chose_high_oriented_to_high.shape[0]
    # plot
    fig, ax = plt.subplots(constrained_layout=True)
    x = np.arange(0,1,0.5)
    labels = ('oriented to low', 'oriented to high')
    measurements = [proportion_chose_high_oriented_to_low, 
                    proportion_chose_high_oriented_to_high]
    ax.bar(x, measurements, alpha=0.75, width=0.3, label=labels)
    plt.xticks(x, labels)
    plt.ylabel('P(Choose High)')
    plt.title('Probability of choosing high depending on head orientation angle')
    plt.show()
 

