import sleap
import numpy as np
import math
import matplotlib.pyplot as plt
from find_frames import access_metadata, relevant_session_frames
import utils.manipulate_data.data_filtering as data_filtering
import cv2
from head_angle import get_wall_angles, extract_head_angle_trial, color_video_frame
from utils.manipulate_data.normalising import normalise
from utils.get_values.get_trial_walls import get_trial_walls


def distance_to_wall_session(octpy_metadata, frames, labels_file):
    """ Find the normalised distance to each wall at given frame for
        one session.
        Where trail is choiceLG or choiceGL, wall_1 is high, wall_2 is low"""
    distance_wall_1 = []
    distance_wall_2 = []
    for trial in range(len(frames)):
        walls = octpy_metadata.iloc[trial].wall
        trial_type = octpy_metadata.iloc[trial].trial_type
        walls = get_trial_walls(trial_type, walls)
        distance_wall_1.append(distance_to_wall(trial, frames, labels_file, walls[0]))
        distance_wall_2.append(distance_to_wall(trial, frames, labels_file, walls[1]))

    return distance_wall_1, distance_wall_2

def distance_to_wall(trial, frames, labels_file, wall):
    """ find the distance from the subject to the wall for specified wall
        Returns the normalised distance for the specified trial """
    # index frames at trial
    frame_idx = frames.iloc[trial]
    
    # index labels_file at frame
    labels = labels_file
    labeledFrames = labels.labeled_frames
    labeledFrame = labeledFrames[frame_idx]

    # get track
    try:
        track0 = labeledFrame.instances[0]
    except IndexError:
        print("Less than expected number of instances in frame")

    # extract points
    points = track0.points
    BodyUpper = points[4]
    subject_coords = (BodyUpper.x, BodyUpper.y, BodyUpper.visible)

    # account for either point not being present
    if not BodyUpper.visible:
        print("One or more points are invisible in frame.")
        return None

    # wall coordinates
    # starting from rightmost wall and going clockwise
    wall_y_coords = [547, 205, 63, 208, 549, 885, 1023, 884]
    wall_x_coords = [1177, 1041, 698, 360, 220, 363, 700, 1036]


    # subtract y coordinates from ymax to find coords in image convention
    wall_y_coords = np.array(wall_y_coords)
    wall_y_coords = 1080 - wall_y_coords
    wall_y_coords = list(wall_y_coords)
    
    wall_xcoord, wall_ycoord = wall_x_coords[wall - 1], wall_y_coords[wall - 1]

    # normalise distance using centre and top-left coordinates
    # variables
    centre = np.array((699, 532))
    top_left = np.array((227, 66))
    wall = np.array([wall_xcoord, wall_ycoord])
    subject = np.array([subject_coords[0], subject_coords[1]])
    half_arena = centre - top_left

    # # TEST
    # # should return subject always ~0.5 distance from wall
    # subject = centre

    normalised_wall = normalise(wall[0], wall[1])
    normalised_subject = normalise(subject[0], subject[1])
    
    # find normalised distance between wall and body
    # find x and y differences for pythagoras
    x_diff = abs(normalised_wall[0] - normalised_subject[0])
    y_diff = abs(normalised_wall[1] - normalised_subject[1])

    # pythagoras 
    normalised_distance = np.sqrt(x_diff**2 + y_diff**2)

    return normalised_distance







if __name__ == '__main__':
    """ return distances to wall 1 and wall 2 for a session. Also return distance to wall 1 at stim_start of single trial
        along with colorcam frame at this point for checking """
    labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02_A006_full_predictions_model5_CLI.slp'
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

    tracks = np.zeros(greyStims_choice.shape[0])
    # head angle at stim_start of specified trial
    angle, vector = extract_head_angle_trial(trial=trial, stim_frames=greyStims_choice, labels_file=labels_file, track_num=0, plotFlag=False)
    print(f"Head direction angle at the start of trial {trial} is {angle:.2f} rad.")

    # show color video frame at stim_start of specified trial 
    file_path = color_video_frame(color_video_path, colorTrials_stim, trial=trial, color_delay=color_delay)

    # find the walls for trial
    trial_type = octpy_metadata_choice.iloc[trial].trial_type
    walls = octpy_metadata_choice.iloc[trial].wall
    walls = get_trial_walls(trial_type, walls)
    normalised_distance = distance_to_wall(trial, greyStims_choice, labels_file, walls[0])
    print(f"Walls for this trial are {walls}, and the normalised distance to the high wall is {normalised_distance:.2f}")

    # find distance to walls for all trials in one session
    # wall_1 is high, wall_2 is low
    distances_wall_1, distances_wall_2 = distance_to_wall_session(octpy_metadata_choice, greyStims_choice, labels_file)

