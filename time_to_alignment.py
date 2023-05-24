import math
import numpy as np



# currently threshold allows 22.5 degrees (pi/8) either side of wall 
# centre
# This is a total 90 degree cone centered on the wall for trial where
# distance between walls == 0
THRESHOLD = math.pi/8
GREYSCALE_FRAMERATE = 50

def time_to_alignment_trial(head_angle_wall_1, head_angle_wall_2):
    """ find time from stim_on until head_angle roughly aligns with
     stimulus walls
      
    Return time, or np.nan if N/A """
    
    threshold = THRESHOLD
    greyscale_framerate = GREYSCALE_FRAMERATE
    
    # head_angles_to_wall_1 and head_angles_to_wall_2

    # when head_angles_to_wall_1 or head_angles_to_wall2 < threshold
    idxs_wall_1_ang = np.where(head_angle_wall_1 < threshold)
    idxs_wall_2_ang = np.where(head_angle_wall_2 < threshold)
    
    # record current frame number in trial
    # account for list being empty
    try:
        first_frame = min(idxs_wall_1_ang[0][0], idxs_wall_2_ang[0][0])
    except IndexError:
        try:
            first_frame = idxs_wall_1_ang[0][0]
        except IndexError:
            try:
                first_frame = idxs_wall_2_ang[0][0]
            # account for both lists being empty
            except IndexError:
                return np.nan
    # based on framerate, return time until this occurs
    time = (1/greyscale_framerate) * first_frame

    return time

def time_to_alignment_session(head_angle_wall_1_session, head_angle_wall_2_session):
    """ repeat the trial function for a full session """

    times = []
    for trial in range(len(head_angle_wall_1_session)):
        times.append(time_to_alignment_trial(head_angle_wall_1_session[trial],
                                             head_angle_wall_2_session[trial]))
    
    return times