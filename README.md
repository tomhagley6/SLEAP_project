Smoothing trajectories: Smoothing uses a 1D Savitzky-Golay filter. Specify smoothing as a parameter when extracting
trajectories in a session or directly smooth a single trial from the pd.Dataframe of session extracted trajectories
Smoothing functions can be found in utils.smooth_trajectory 

Extracting head angles: Extract head angle from sleap pose data as the angle of the vector from neck to nose 
relative to the horizontal (going CCW). 
Functions can be found in head_angle.py, with functions to extract head angle at specified frames in each trial
(e.g. stim start frames or trial end frames),or just a single specified frame index.  
Functions take track number as an input, so must run separately on each track in social sessions.

logistic_regression.py: makes use of head_angle.py function head_angle_to_wall to find the head angles to
both of the walls, record them in om, and compare them. This ONLY works for social sessions, as it assumes
track 0 in the sleap trajectory data# SLEAP_project
