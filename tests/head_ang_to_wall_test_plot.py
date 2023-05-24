from session import Session
from utils.plotting.trial_start_test_plot_head_ang_to_wall import trial_start_test_plot_head_ang_to_wall
import trajectory_extraction
import head_angle
from utils.node_coordinates import node_coordinates_at_frame
from utils.head_angle_to_wall import head_angle_to_wall
from utils.get_values.get_wall_angles import get_wall_angles
import math

""" For one session, test the head_angle_to_wall extraction """



root = '/home/tomhagley/Documents/SLEAPProject'
session = '2022-11-04_A004'
sess = Session(root, session)
sess.extract_basic_data()
sess.more_complex_extraction()


# get continuous trajectory for full video (allows indexing by frame_num)
# video_trajectories, column_names = trajectory_extraction.extract_video_trajectory(sess.trajectories_path, normalise=True, smooth=True)
for trial in [1]:
        # get the head angle for a single frame
        head_ang, head_vector = head_angle.extract_head_angle_frame(sess.grey_stims[trial], sess.labels_file, track_num=0, plotFlag=False) #1050
        # get the (neck) node coordinates for a single frame
        neck_coords = node_coordinates_at_frame(sess.grey_stims[trial], 'neck', sess.video_trajectories)
        angle_to_wall_high = head_angle_to_wall(head_vector, sess.wall_1_session[trial], neck_coords)
        angle_to_wall_low = head_angle_to_wall(head_vector, sess.wall_2_session[trial], neck_coords)

        ### PRINT RESULTS ###
        print(f"Head angle is {head_ang:.2f}, or {math.degrees(head_ang):.2f} degrees")
        print(f"High wall number is {sess.wall_1_session[trial]}\nLow wall number is {sess.wall_2_session[trial]}")
        wall_angles = get_wall_angles()
        print(f"Angle of High wall to horizontal is {wall_angles[sess.wall_1_session[trial] - 1]:.2f}, or {math.degrees(wall_angles[sess.wall_1_session[trial] - 1]):.2f} degrees")
        print(f"Angle of Low wall to horizontal is {wall_angles[sess.wall_2_session[trial] - 1]:.2f}, or {math.degrees(wall_angles[sess.wall_2_session[trial] - 1]):.2f} degrees")
        print(f"Neck coords are {neck_coords[0]:.2f}, {neck_coords[1]:.2f}")
        print(f"Angle to High wall is {angle_to_wall_high:.2f}, or {math.degrees(angle_to_wall_high):.2f} degrees")
        print(f"Angle to Low wall is {angle_to_wall_low:.2f}, or {math.degrees(angle_to_wall_low):.2f} degrees")



        trial_start_test_plot_head_ang_to_wall(sess.octpy_metadata, trial, sess.grey_stims, sess.labels_file,
                                               0, sess, plotFlag=True)
        



