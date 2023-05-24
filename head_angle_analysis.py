import numpy as np
import math
import matplotlib.pyplot as plt
from find_frames import access_metadata, relevant_session_frames
import utils.manipulate_data.data_filtering as data_filtering
from head_angle import *


""" collection of analyses based on head direction angle """


# pd apply function
# Assumes grating is high!
def order_walls(walls, trial_type):
    """ reverse order of walls in OctPy if trial_type is LG """
    intWalls = [int(x) for x in walls.split(',')]
    if trial_type == 'choiceLG':
        intWalls.reverse()
    
    return intWalls

# pd apply function
# OLD, this is more complex and handled in the utils function
def smallest_head_angle_to_wall_OLD(head_angle, wall_angle):
    """ find the obtuse angle between the head and the wall """
    difference = abs(head_angle - wall_angle)
    if difference > math.pi:
        difference = 2*math.pi - difference
    
    return difference


def head_angle_to_wall_stim_on(octpy_metadata, stim_frames, labels_file, wall_angles, track_num, checks=False, check_trial=10):
    """ Update octpy metadata to include head angle to the walls """

    om = octpy_metadata

    # get stim start head angles for the session
    om['head_angle_stim'] = extract_head_angle_session(stim_frames, labels_file, track_num=track_num, plotFlag=False)
    
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
        print(f"Angle between the two walls is {abs(om_checks['head_ang_wall_high'] - om_checks['head_ang_wall_low']):.3f}")
        print(f"Given these two values, 'ang_close_to_low' == {om_checks['ang_close_to_low']}")

    return om

# UNUSED
def head_angle_to_wall_relative_diff(wall_high_ang, wall_low_ang, wall_high_head_ang, wall_low_head_ang, ang_close_to_low):
    """ Define relative difference between head-wall angles as 
        ang_between_walls/head_ang_to_centre_of_walls
        Negative if closer to low, positive if closer to high """
    
    # paramas
    wall_angles = get_wall_angles()
    trial_wall_angles = [wall_high_ang, wall_low_ang]
    
    # find the angle between walls as the difference between the two wall angles
    # also find the angle directly between the two walls
    # account for being a circular statistic by adding 2pi in the special case
    if wall_angles[0] in trial_wall_angles and wall_angles[1] in trial_wall_angles:
        ang_between_walls = abs(wall_angles[1] - (wall_angles[0] + 2*math.pi))
        centre_wall_ang = wall_angles[1] + ang_between_walls
    else:
        ang_between_walls = abs(wall_high_ang - wall_low_ang)
        centre_wall_ang = min(wall_high_ang, wall_low_ang) + ang_between_walls / 2
    
    # find the head angle to the centre of the two walls by adding half 
    # the angle between walls to the angle to the closest wall
    head_ang_to_closest_wall = min(wall_high_head_ang, wall_low_head_ang)
    head_ang_to_centre_of_walls = head_ang_to_closest_wall + 0.5*ang_between_walls

    relative_diff = ang_between_walls / head_ang_to_centre_of_walls
    
    # negative if closest to low, positive if closest to high
    if ang_close_to_low == True:
        relative_diff = -(relative_diff)


    head_ang_to_centre_of_walls = wall_high_head_ang + wall_low_head_ang
    relative_diff = ang_between_walls / centre_wall_ang

    return None

 ## TESTING ONLY
if __name__ == '__main__':
    """ Show bar chart for P(high) against head ang_close_to_low """
    # params
    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    session = '2022-11-02_A006'
    labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.slp'
    labels_file = sleap.load_file(labelsPath)
    wall_angles = get_wall_angles()
    trial = 20
    color_delay = 52

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
 