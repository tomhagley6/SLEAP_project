import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
import sleap
import cv2 
import re

refreshFiles = False
###  find video frames for a trial event ###

# paths
video_path = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraTop_2022-11-02T14-00-00.avi'

directory_octpy = '/home/tomhagley/Documents/SLEAPProject/data/bonsai_metadata'
fileName_octpy = '2022-11-02.csv'
octpy_path = directory_octpy + os.sep + fileName_octpy

directory_sleap = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
fileName_sleap = 'model5_predictions_230206_CLI._CameraTop_2022-11-02T14-00-00.analysis.h5'
sleap_path = directory_sleap + os.sep + fileName_sleap

# aeon bonsai data
if refreshFiles:
    # transfer from hpc
    os.system(f" scp -r  thagley@ssh.swc.ucl.ac.uk:~/files/data/octagon/aeon_data /home/tomhagley/Documents/SLEAPProject/data/")

# load df
videoMetadata = (pd.read_csv('/home/tomhagley/Documents/SLEAPProject/data/aeon_data/video_2022-11-02T15-30-47.csv')
                 .astype({'time':'datetime64'})
                 .set_index('time')
                 )
colorvideoMetadata = (pd.read_csv('/home/tomhagley/Documents/SLEAPProject/data/aeon_data/colorvideo_2022-11-02T15-30-47.csv')
                      .astype({'time':'datetime64'})
                      .set_index('time')
                     )

# octpy metadata
if refreshFiles:
    # transfer metadata files from hpc
    os.system(f" scp -r  thagley@ssh.swc.ucl.ac.uk:~/files/data/octagon/bonsai_metadata /home/tomhagley/Documents/SLEAPProject/data/")

# load df
octpyMetadata = (pd.read_csv(octpy_path)
            .astype({'time':'datetime64'})
            .set_index('time')
            )

# relevant trial frames
# params
trial = 10
framerate_greyscale = 50
# trial_start
trialStart_time = octpyMetadata.index.values[trial]
# stim_start
stimDelay = octpyMetadata.iloc[trial].stimulus_delay
stimDelay_frames = stimDelay*framerate_greyscale
stimStart_time = trialStart_time + pd.Timedelta(stimDelay, 's') # need to use pandas timedelta if passing a float 
# frame index for stim_start
stimStart_idx = videoMetadata.index.get_loc(stimStart_time, method='ffill')
stimStart_frame = videoMetadata.iloc[stimStart_idx]
stimStart_frameNum = stimStart_frame._frame


# trial_end
# Remember RT is from trial_onset, not stim_onset
naiveRT = octpyMetadata.iloc[trial].RT
RT = pd.Timedelta(naiveRT, 's') - (stimStart_time - trialStart_time)
RT = RT/pd.Timedelta(1,'s') # convert from Timedelta to seconds float
RT_frames = math.ceil(RT*framerate_greyscale)


timeStamp = "2022-11-02 14:33:00"
timeSlice = octpyMetadata.loc[: '2022-11-02 14:30:40']

# find colorvideo framerate
startIdx = 0
startTimestamp = colorvideoMetadata.index.values[0]
endTimestamp = startTimestamp + pd.Timedelta(1, 's')
# framerate is idx of nearest timestamp to startTime + 1 s
endIdx = colorvideoMetadata.index.get_loc(endTimestamp, method= 'nearest')
framerate_color = endIdx

# find stim_on frame in colorvideo
stimStart_idx_color = colorvideoMetadata.index.get_loc(stimStart_time, method='ffill')
trialStart_idx_color = colorvideoMetadata.index.get_loc(trialStart_time, method='ffill')
trialStart_time_color = colorvideoMetadata.index.values[trialStart_idx_color]


# load colorcamera video and define clip 
cap = cv2.VideoCapture('/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraColorTop_2022-11-02T14-00-00.avi')
start, end = trialStart_idx_color, trialStart_idx_color + framerate_greyscale*10
parts = [(start, end)]
ret, frame = cap.read()
h, w, _ = frame.shape

# define writers for writing video clip
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writers = [cv2.VideoWriter(f"part{start}-{end}.avi", fourcc, 60.0, (w, h)) for start, end in parts]

# for each clip, loop through all frames and only write the ones within clip indexes
f = 0
while ret:
    f += 1
    for i, part in enumerate(parts):
        start, end = part
        if start <= f <= end:
            writers[i].write(frame)
    ret, frame = cap.read()

# release writers and capture
for writer in writers:
    writer.release()

cap.release()

# now play videos

# play trial video with opencv
#cap = cv2.VideoCapture('/home/tomhagley/Documents/SLEAPProject/octagon_solo/analysis/part9756-9956.avi')
cap = cv2.VideoCapture('/home/tomhagley/Documents/SLEAPProject/octagon_solo/analysis/part8880-9380.avi')
# Read until video is completed
while(cap.isOpened()):
    
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1)
    # Press Q on keyboard to exit
        # if cv.waitKey(25) & 0xFF == ord('q'):
        if key == ord('q'):
            break
    # Press P on keyboard to pause
        if key == ord('p'):
            cv2.waitKey(-1) # wait for any key press

    else:
        break

    cv2.waitKey(int(25))

# When done, release video capture obj
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# TODO
def relevant_frames_from_session(videoMetadata, octpyMetadata):
    """ Return trial_start, stim_start, and response frames 
        for each trial in a session """
    return None

# TODO
def access_metadata(root, session, colorVideo=False):
    """ Session in form {year}-{month}-{day}T{hour}-{min}-{seconds}
        Session is used to find octpy and video metadata files """
    
    # paths
    video_path = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraTop_2022-11-02T14-00-00.avi'

    directory_octpy = '/home/tomhagley/Documents/SLEAPProject/data/bonsai_metadata'
    fileName_octpy = '2022-11-02_A006.csv'
    octpy_path = directory_octpy + os.sep + fileName_octpy

    directory_sleap = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
    fileName_sleap = 'model5_predictions_230206_CLI._CameraTop_2022-11-02T14-00-00.analysis.h5'
    sleap_path = directory_sleap + os.sep + fileName_sleap

    # aeon bonsai find_frames_cleanata
    if refreshFiles:
        # transfer from hpc
        os.system(f" scp -r  thagley@ssh.swc.ucl.ac.uk:~/files/data/octagon/aeon_data /home/tomhagley/Documents/SLEAPProject/data/")

    filePath = root + os.sep + 'aeon_mecha'
    pattern = re.compile(r"^video_" + session + r"T\d\d-\d\d-\d\d.csv")
    files = [re.findall(pattern, fileName)[0] for fileName in filePath if re.findall(pattern, fileName)]

    # load metadata for all files in session
    videoMetadata_list = []
    for file in files:
        videoMetadata = (pd.read_csv(f'/home/tomhagley/Documents/SLEAPProject/data/aeon_data/{file}')
                        .astype({'time':'datetime64'})
                        .set_index('time')
                        )
        videoMetadata_list.append(videoMetadata)

    if colorVideo:
        colorvideoMetadata = (pd.read_csv('/home/tomhagley/Documents/SLEAPProject/data/aeon_data/colorvideo_2022-11-02T15-30-47.csv')
                            .astype({'time':'datetime64'})
                            .set_index('time')
                            )

    # octpy metadata
    if refreshFiles:
        # transfer metadata files from hpc
        os.system(f" scp -r  thagley@ssh.swc.ucl.ac.uk:~/files/data/octagon/bonsai_metadata /home/tomhagley/Documents/SLEAPProject/data/")

    # load df
    octpyMetadata = (pd.read_csv(octpy_path)
                .astype({'time':'datetime64'})
                .set_index('time')
                )

    return octpyMetadata, videoMetadata

def access_video_metadata(root, session, colorVideo=False, refreshFiles=False):
    
    # aeon bonsai data
    if refreshFiles:
        # transfer from hpc
        os.system(f" scp -r  thagley@ssh.swc.ucl.ac.uk:~/files/data/octagon/aeon_data /home/tomhagley/Documents/SLEAPProject/data/")

    filePath = root + os.sep + 'aeon_mecha'
    pattern = re.compile(r"^video_" + session + r"T\d\d-\d\d-\d\d.csv")
    files = [re.findall(pattern, fileName)[0] for fileName in filePath if re.findall(pattern, fileName)]

    # load metadata for all files in session
    videoMetadata_list = []
    for file in files:
        videoMetadata = (pd.read_csv(f'/home/tomhagley/Documents/SLEAPProject/data/aeon_data/{file}')
                        .astype({'time':'datetime64'})
                        .set_index('time')
                        )
        videoMetadata_list.append(videoMetadata)

    if colorVideo:
        colorvideoMetadata = (pd.read_csv('/home/tomhagley/Documents/SLEAPProject/data/aeon_data/colorvideo_2022-11-02T15-30-47.csv')
                            .astype({'time':'datetime64'})
                            .set_index('time')
                            )

    return videoMetadata_list


# TODO
def save_video(root, session, trial, colorVideo=False):
    """ Save video clip (reuse code here)"""
    return None

# TODO
def play_video(root, session, trial, colorVideo=False):
    """ Playback video clip (reuse code here)
        Call this as optional in save_video """
    
# TODO
# extract head angle from start frames, compare to response, visualise

# if __name__ == '__main__':
#     video_path = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraTop_2022-11-02T14-00-00.avi'

#     directory_octpy = '/home/tomhagley/Documents/SLEAPProject/data/bonsai_metadata'
#     fileName_octpy = '2022-11-02_A006.csv'
#     octpy_path = directory_octpy + os.sep + fileName_octpy

#     directory_sleap = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
#     fileName_sleap = 'model5_predictions_230206_CLI._CameraTop_2022-11-02T14-00-00.analysis.h5'
#     sleap_path = directory_sleap + os.sep + fileName_sleap


#     data_root = '/home/tomhagley/Documents/SLEAPProject/data'
#     data_type = 'aeon_mecha'
#     session = '2022-11-02'
    
#     videoMetadata_list = access_video_metadata(root=data_root, session=session)
#     print(videoMetadata_list)

#     # pattern = re.compile(r"^video_" + session + r"T\d\d-\d\d-\d\d.csv")
#     # files = [re.findall(pattern, fileName)[0] for fileName in filePath if re.findall(pattern, fileName)]