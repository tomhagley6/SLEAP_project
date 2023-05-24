Overview of common functions in general order of use:

find_frames.access_metadata:
assumes video and session metadata can be found on the hpc, then moves this data to the local machine
and loads it into memory

data_filtering:
collection of functions that filters trial metadata based on e.g. response time, trial type, or change of mind 

find_frames.relevant_frames_from_session_vectorised:
uses trial metadata to find frames for trial start, stim start, and trial end, in both cameras

utils.correct_color_camera_frame_offset:
contains function to find the color camera frame offset either at the beginning or end of the session (this is necessary for octagon 1 as color camera timestamp is shifted relative to greyscale camera)

trajectory_extraction.extract_session_trajectories:
from sleap data, find trajectories for each node, for each trial in the session
trajectories are always interpolated to fill NaNs, but can also be smoothed, flipped_and_rotated, and normalised

trajectory_extraction.extract_video_trajectories:
same as above, but using the full video instead of any trial structure (can find trajectories at arbitrary frame numbers)

head_angle.extract_head_angle_session:
find the (specified) subject head angle relative to horizontal at specified frames (intended to be the output of relevant_frames_from_session)

utils.get_values.get_wall_numbers:
function to return the wall numbers of wall_1 (normally high) and wall_2 (normally low) for a session

utils.get_values.get_head_to_wall_angles:
function to get the smallest angle between head and wall_1 and wall_2 for each trial in the session, at specified frames (intended to be output of relevant_frames_from_session)

head_angle_to_other:
TODO

utils.distance_to_wall:
functions to find the distance between the mouse and each wall at the START of a trial

utils.distance_to_wall_frame:
TODO

find_frames.timestamps_within_trial:
return a list of timestamps of all greyscale video frames in each trial in a session

utils.trajectory_speeds:
function to extract speeds for all trajectories of all trials in a session, with smoothing and filtering

utils.change_of_mind:
functions to identify change of mind trial, and apply this to a full session. This is currently based on maintaining head_to_wall_angle below a specified threshold for a specified number of consecutive frames within the trial for BOTH walls

utils.change_of_mind.save_CoM_videos:
save color videos of full trials given by CoM_trial indexes
Can make this a more general function

logistics_regression:
function and script to run a logistic regression on independent variables including distance to either wall and head angle to either wall

Less common functions:

utils.get_values.get_node_numbers:
for a given node name, return index of node in skeleton

utils.node_coordinates:
functions to get the coordinates of a given skeleton node at any specified frame or a series of specified frames in the session

utils.get_values.get_wall_coords: 
return x-y coords for a given wall numbers (rightward wall 1, continue clockwise)

utils.normalising:
normalise a single point, or use np.vectorize and normalise a 1D vector of point x/y coordinates

utils.pandas_apply_functions:
any small functions for use with pd.apply

utils.video_functions.save_video_trial:
save a video clip, either a SLEAP annotated video, or directly from the video file. 
Intended to to save individual trials, so requires stim_start and trial_end frame numbers 

utils.angle_between_vectors:
function to find angle between two vectors - used by other functions

utils.flip_and_rotate_trajectory:
all functions needed to flip and rotate trajectories s.t high wall is at wall 7, and low wall is at wall 8. Currently only works for trials with walls separation of 45 degrees

OTHER:

sleap_visualisation:
input a trial number (and paths for current session) and run all cells. Will return colour video frame at trial start (corrected from greyscale timestamp and with 5 frames to account for projector lag), annotated sleap frames at trial start and trial end, and greyscale camera video of full trial

tests:
tests for more complex utility functions

extraction.py:
script sandbox for playing with sleap exported data

misc scripts: 
scripts for doing basic analysis of different trial variables