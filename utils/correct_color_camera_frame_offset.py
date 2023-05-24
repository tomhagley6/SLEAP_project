import numpy as np
import pandas as pd

FRAMERATE_COLOR_CAMERA = 60

def find_color_camera_frame_offset(video_metadata, color_video_metadata, framerate_color_camera=FRAMERATE_COLOR_CAMERA, start=True):
    if start:
        idx_grey, idx_color  = 0, 0
    else:
        idx_grey, idx_color = video_metadata.shape[0] - 1, color_video_metadata.shape[0] - 1
    # find the difference in start timestamps between the two cameras
    time_diff = abs((video_metadata.index.values[idx_grey] - color_video_metadata.index.values[idx_color]) / np.timedelta64(1, 's'))
    # convert this difference into a frame number
    color_frame_delay = int(time_diff * framerate_color_camera)
    
    
    return color_frame_delay

def find_color_camera_frame_offset_trial(video_metadata, color_video_metadata, grey_starts,
                                          color_starts, trial, framerate_color_camera=FRAMERATE_COLOR_CAMERA):


    idx_grey = grey_starts[trial]
    # find nearest color_video_camera timestamp to the grey timestamp
    grey_timestamp = grey_starts.index.values[trial]
    color_timestamp = color_starts.index.values[trial]
    # nearest_color_timestamp_idx = color_video_metadata.index.get_loc(grey_timestamp, method= 'nearest')
    # nearest_color_timestamp = color_video_metadata.index.values[nearest_color_timestamp_idx]

    idx_grey, idx_color  = grey_starts[trial], color_starts[trial]

    # find the difference in start timestamps between the two cameras
    time_diff = abs((grey_timestamp - color_timestamp) / np.timedelta64(1, 's'))
    # convert this difference into a frame number
    color_frame_delay = int(time_diff * framerate_color_camera)

    return color_frame_delay