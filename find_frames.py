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
from utils.correct_video_frame_reset import correct_video_frame_reset

FRAMERATE_GREYSCALE = 50


def relevant_session_frames(octpy_metadata, video_metadata, color_video_metadata):
    """ Return trial_start, stim_start, and response frames for each trial in a session
        Do this for colour camera and greyscale camera
         
          Input:
            octpy_metadata - trial metadata from octpy, returned by access_metadata
             video_metadata - camera frame data, returned by access_metadata
            color_video_metadata - as above """

    # # parameters
    # datasets
    vm = video_metadata
    cvm = color_video_metadata

    om = octpy_metadata

    framerate_greyscale = FRAMERATE_GREYSCALE
    
    # trial_start times
    # default index of om 
    trial_start_times = pd.Series(om.index.values)
    
    # stim_start times
    # stim start is the trial start plus the stimulus delay
    om['stim_start_time'] = om['ts'] + om['stimulus_delay'].apply(pd.to_timedelta, unit='S')
    stim_start_times = om.reset_index().stim_start_time # replace trial_start_time index with ordinal index

    # account for the camera frame_num reset that occurs when concatenating multiple videos
    camera_reset = vm[vm._frame == 0]
    vm = correct_video_frame_reset(vm)
    cvm = correct_video_frame_reset(cvm)

    # vm_corrected = pd.Series(vm._frame.copy().values, index=np.arange(vm._frame.values.shape[0]))
    # frame_reset = vm_corrected[vm_corrected == 0].index.values[-1]
    # if frame_reset != 0:
    #     vm_corrected[frame_reset:] = vm_corrected[frame_reset:] + frame_reset
    #     vm._frame = vm_corrected.values
    
    # # Greyscale video
    # stim start
    # use merge_asof instead of index.get_loc for vectorised operation (must be sorted by the key column)
    # returns a left-join merge of the two dataframes, but can extract the column that gives us vm camera frame times
    # left_on and right_on are redundant as we are passing single-column df
    stim_start_vmTimes = pd.merge_asof(om.stim_start_time, vm.reset_index(level=0).time, left_on='stim_start_time', right_on='time', direction='backward')
    stim_start_vmTimes = stim_start_vmTimes.time
    stim_start_vm = vm.loc[list(stim_start_vmTimes)]
    stim_start_frame_nums = stim_start_vm._frame # frame numbers at stim start 

    # trial start
    trial_start_vmTimes = pd.merge_asof(om.ts, vm.reset_index(level=0).time, left_on='ts', right_on='time', direction='backward')
    trial_start_vmTimes = trial_start_vmTimes.time
    trial_start_vm = vm.loc[list(trial_start_vmTimes)]
    trial_start_frame_nums = trial_start_vm._frame # frame numbers at trial start

    # trial end
    naiveRTx = om.reset_index().RT
    RTx = naiveRTx.apply(pd.to_timedelta, unit='S') - (stim_start_times - trial_start_times)
    RTx_float = RTx/pd.Timedelta(1,'s')
    RT_framesx = np.ceil(RTx_float*framerate_greyscale)
    trial_end_frame_nums_naive = stim_start_frame_nums.reset_index()._frame + RT_framesx.astype('int64') # can only add series if they have compatible/no index
    trial_end_frame_nums_naive = trial_end_frame_nums_naive.rename('trial_end_frame_nums_naive')
    trial_end_times = stim_start_times + RTx 
    trial_end_times = trial_end_times.rename('trial_end_times') # cannot merge a series without a name
    
    # careful using 'forward' for matching between series
    # use 'nearest' match if 'forward' fails
    # try match forward here because we want to be inclusive of trial end
    try:
        trial_end_vmTimes = pd.merge_asof(trial_end_times, vm.reset_index(level=0).time, 
                                          left_on='trial_end_times', right_on='time', 
                                          direction='forward')
        trial_end_vmTimes = trial_end_vmTimes.time
        trial_end_vm = vm.loc[list(trial_end_vmTimes)]
        trial_end_frame_nums = trial_end_vm._frame # frame numbers at trial end
    
    except KeyError: # occurs when merge_asof has no timestamp to merge forward to
        trial_end_vmTimes = pd.merge_asof(trial_end_times, vm.reset_index(level=0).time, 
                                          left_on='trial_end_times', right_on='time', 
                                          direction='nearest')
        trial_end_vmTimes = trial_end_vmTimes.time
        trial_end_vm = vm.loc[list(trial_end_vmTimes)]
        trial_end_frame_nums = trial_end_vm._frame # frame numbers at trial end

    # # Colour video
    # repeat the above but for colour video data
    # find  framerate
    startTimestamp = cvm.index.values[0]
    endTimestamp = startTimestamp + pd.Timedelta(1, 's')
    # framerate is idx of nearest timestamp to startTime + 1 s
    endIdx = cvm.index.get_loc(endTimestamp, method= 'nearest')
    framerate_color = endIdx

    # stim start
    stim_start_vmTimes_color = pd.merge_asof(om.stim_start_time, cvm.reset_index(level=0).time, 
                                             left_on='stim_start_time', right_on='time', 
                                             direction='backward')
    stim_start_vmTimes_color = stim_start_vmTimes_color.time
    stim_start_vm_color = cvm.loc[list(stim_start_vmTimes_color)]
    stim_start_frame_nums_color = stim_start_vm_color._frame

    # trial start
    trial_start_vmTimes_color = pd.merge_asof(om.ts, vm.reset_index(level=0).time, 
                                              left_on='ts', right_on='time', 
                                              direction='backward')
    trial_start_vmTimes_color = trial_start_vmTimes_color.time
    trial_start_vm_color = vm.loc[list(trial_start_vmTimes_color)]
    trial_start_frame_nums_color = trial_start_vm_color._frame

    # trial end
    # careful using 'forward' for matching between series
    # use 'nearest' match if 'forward' fails
    try: 
        
        RT_framesx_color = np.ceil(RTx_float*framerate_color)
        trial_end_frame_nums_color_naive = stim_start_frame_nums + RT_framesx_color
        # match forward here because we want to be inclusive of trial end
        trial_end_vmTimes_color = pd.merge_asof(trial_end_times, cvm.reset_index(level=0).time, 
                                                left_on='trial_end_times', right_on='time', direction='forward')
        trial_end_vmTimes_color = trial_end_vmTimes_color.time
        trial_end_vm_color = cvm.loc[list(trial_end_vmTimes_color)]
        trial_end_frame_nums_color = trial_end_vm_color._frame

    except KeyError: # occurs when merge_asof has no timestamp to merge forward to
        
        RT_framesx_color = np.ceil(RTx_float*framerate_color)
        trial_end_frame_nums_color_naive = stim_start_frame_nums + RT_framesx_color
        # match forward here because we want to be inclusive of trial end
        trial_end_vmTimes_color = pd.merge_asof(trial_end_times, cvm.reset_index(level=0).time, 
                                                left_on='trial_end_times', right_on='time', direction='nearest')
        trial_end_vmTimes_color = trial_end_vmTimes_color.time
        trial_end_vm_color = cvm.loc[list(trial_end_vmTimes_color)]
        trial_end_frame_nums_color = trial_end_vm_color._frame

    # correct for camera frame number resetting on the hour
    # this is useful for being able to index sleap trajectory data for the full concatenated video
    # data of the session
    # trial_start_frame_nums, stim_start_frame_nums, trial_end_frame_nums \
    #       = correct_camera_frame_reset(trial_start_frame_nums, stim_start_frame_nums, trial_end_frame_nums)
    # trial_start_frame_nums_color, stim_start_frame_nums_color, trial_end_frame_nums_color \
    #       = correct_camera_frame_reset(trial_start_frame_nums_color, stim_start_frame_nums_color, trial_end_frame_nums_color)
    

    return trial_start_frame_nums, stim_start_frame_nums, trial_end_frame_nums, trial_start_frame_nums_color, stim_start_frame_nums_color, trial_end_frame_nums_color

def timestamps_within_trial(stim_frames, end_frames, video_metadata):
    """ Return a list of timestamps for all frames
        in each trial (list entries) for a session
        
        Input:
            stim_frames - stimulus onset frames from relevant_session_frames
            end_frames - trial end frames from relevant_session_frames
            video_metadata - video data from access_metadata"""

    # loop each trial and find timestamps from video data
    timestamps_list = []
    for trial in range(stim_frames.shape[0]):
        start_frame, end_frame = stim_frames[trial], end_frames[trial]
        relevant_frames = video_metadata[start_frame:end_frame]
        relevant_frames_timestamps = relevant_frames.index.values
        timestamps_list.append(relevant_frames_timestamps)

    return timestamps_list

## UNUSED
def relevant_trial_frames(octpyMetadata, videoMetadata_list, colorVideoMetadata_list, trial):
    """ Return trial_start, stim_start, and response frames for a single trial in a session """
    
    trial = trial

    # only working for one video - metadata list len 1
    videoMetadata = videoMetadata_list
    colorVideoMetadata = colorVideoMetadata_list

    # relevant trial frames
    # params
    framerate_greyscale = 50
    # trial_start
    trial_start_time = octpyMetadata.index.values[trial]
    # stim_start
    stimDelay = octpyMetadata.iloc[trial].stimulus_delay
    stimDelay_frames = stimDelay*framerate_greyscale
    stim_start_time = trial_start_time + pd.Timedelta(stimDelay, 's') # need to use pandas timedelta if passing a float 
    # frame index for stim_start
    stim_start_idx = videoMetadata.index.get_loc(stim_start_time, method='ffill')
    stim_start_frame = videoMetadata.iloc[stim_start_idx]
    stim_start_frame_num = stim_start_frame._frame
    trial_start_idx = videoMetadata.index.get_loc(trial_start_time, method='ffill')
    trial_start_frame = videoMetadata.iloc[trial_start_idx]
    trial_start_frame_num = trial_start_frame._frame

    # trial_end
    # Remember RT is from trial_onset, not stim_onset
    # Convert RT to num frames after stim_start_time
    naiveRT = octpyMetadata.iloc[trial].RT
    RT = pd.Timedelta(naiveRT, 's') - (stim_start_time - trial_start_time)
    RT = RT/pd.Timedelta(1,'s') # convert from Timedelta to seconds float
    RT_frames = math.ceil(RT*framerate_greyscale)
    trial_end_frame_num = stim_start_frame_num + RT_frames


    timeStamp = "2022-11-02 14:33:00"
    timeSlice = octpyMetadata.loc[: '2022-11-02 14:30:40']

    # find colorvideo framerate
    startIdx = 0
    startTimestamp = colorVideoMetadata.index.values[0]
    endTimestamp = startTimestamp + pd.Timedelta(1, 's')
    # framerate is idx of nearest timestamp to startTime + 1 s
    endIdx = colorVideoMetadata.index.get_loc(endTimestamp, method= 'nearest')
    framerate_color = endIdx

    # find frames in colorvideo
    # stim start
    stim_start_idx_color = colorVideoMetadata.index.get_loc(stim_start_time, method='ffill')
    stim_start_frame_color = colorVideoMetadata.iloc[stim_start_idx_color]
    stim_start_frame_num_color = stim_start_frame_color._frame
    # trial start
    trial_start_idx_color = colorVideoMetadata.index.get_loc(trial_start_time, method='ffill')
    trial_start_time_color = colorVideoMetadata.index.values[trial_start_idx_color]
    trial_start_frame_color = colorVideoMetadata.iloc[trial_start_idx_color]
    trial_start_frame_num_color = trial_start_frame_color._frame
    # trial end 
    RT_frames = math.ceil(RT*framerate_color)
    trial_end_frame_num_color = stim_start_frame_num_color + RT_frames

    return trial_start_frame_num, stim_start_frame_num, trial_end_frame_num, trial_start_frame_num_color, stim_start_frame_num_color, trial_end_frame_num_color

def access_metadata(root, session, colorVideo=False, refreshFiles=False):
    """ Function to find and load trial metadata and camera data files
        flags for which camera and to reload files from HPC 

        Input:
            root - metadata root path
            session - Session in form {year}-{month}-{day}T{hour}-{min}-{seconds}
            
        Output:
            octpy trial metadata, video data, [colour video data] """
    
    octpy_metadata = access_octpy_metadata(root, session, refreshFiles=refreshFiles)
    if colorVideo:
            videoMetadata_list, colorVideoMetadata_list = access_video_metadata(root, session, colorVideo=colorVideo, refreshFiles=refreshFiles)
            return (octpy_metadata, videoMetadata_list, colorVideoMetadata_list)

    else:
        videoMetadata_list = access_video_metadata(root, session, colorVideo=colorVideo, refreshFiles=refreshFiles)
        return (octpy_metadata, videoMetadata_list)

# Function called by access_metadata, not directly accessed
def access_video_metadata(root, session, colorVideo=False, refreshFiles=False):
    """ bonsai metadata from the CameraTop and/or CameraColorTop cameras
        accessed through the aeon octagon API. 
        Session in form {year}-{month}-{day}T{hour}-{min}-{seconds}"""
    
    # aeon bonsai data
    if refreshFiles:
        # transfer from hpc
        os.system(f" scp -r  thagley@ssh.swc.ucl.ac.uk:~/files/data/octagon/aeon_data {root}") # change as needed

    filePath = root + os.sep + 'aeon_data'
    # for grayscale
    # replaced with a new filename convention
    # pattern_video = re.compile(r"^video_" + session[:10] + r"T\d\d-\d\d-\d\d.csv")
    # pattern_video = re.compile(r"^video_" + session[:10] + r"[a-zA-Z0-9_]*.csv")
    pattern_video = re.compile(r"^video_" + session + r".csv")
    files_video = [re.findall(pattern_video, fileName)[0] for fileName in os.listdir(filePath) if re.findall(pattern_video, fileName)]
    # for colour
    # replaced with a new filename convention
    # pattern_colorVideo = re.compile(r"^colorvideo_" + session[:10] + r"T\d\d-\d\d-\d\d.csv")
    # pattern_colorVideo = re.compile(r"^colorvideo_" + session[:10] + r"[a-zA-Z0-9_]*.csv")
    pattern_colorVideo = re.compile(r"^colorvideo_" + session + r".csv")
    files_colorVideo = [re.findall(pattern_colorVideo, fileName)[0] for fileName in os.listdir(filePath) if re.findall(pattern_colorVideo, fileName)]

    # load metadata for all files in session
    videoMetadata_list = []
    for file in files_video:
        fileRoot = f"{root}{os.sep}aeon_data"
        videoMetadata = (pd.read_csv(f'{fileRoot}{os.sep}{file}')
                        .astype({'time':'datetime64'})
                        .set_index('time')
                        )
        videoMetadata_list.append(videoMetadata)

    if colorVideo:
        colorvideoMetadata_list = []
        for file in files_colorVideo:
            fileRoot = f'{root}{os.sep}aeon_data'
            colorvideoMetadata = (pd.read_csv(f'{fileRoot}{os.sep}{file}')
                                .astype({'time':'datetime64'})
                                .set_index('time')
                                )
            colorvideoMetadata_list.append(colorvideoMetadata)

    if colorVideo:
        return videoMetadata_list[0], colorvideoMetadata_list[0]
    else:
        return videoMetadata_list[0]

# Function called by access_metadata, not directly accessed
def access_octpy_metadata(root, session, refreshFiles=False):
    """ Octpy metadata derived from collected octagon bonsai metadata
        accessed through the aeon octagon API. 
        Session in form {year}-{month}-{day}T{hour}-{min}-{seconds}"""
    
    # octpy metadata
    if refreshFiles:
        # transfer metadata files from hpc
        os.system(f" scp -r  thagley@ssh.swc.ucl.ac.uk:~/files/data/octagon/bonsai_metadata {root}") # change as needed

    filePath = root + os.sep + 'bonsai_metadata'
    file = filePath + os.sep + session + '.csv'

    # load df
    # ACCOUNT FOR OCTPY CSV CHANGE
    # octpyMetadata = (pd.read_csv(file)
    #             .astype({'ts':'datetime64'})
    #             .set_index('ts', drop=False)
    #             )
    octpyMetadata = (pd.read_csv(file)
                     .astype({'ts':'datetime64'})
                    )
    octpyMetadata["timestamp"] = octpyMetadata["ts"]
    octpyMetadata = octpyMetadata.set_index('timestamp', drop=True)
    
    return octpyMetadata


def save_video(root, filename, outputPath, stims, ends, labels_filename=None, colorVideo=False, fps=False, color_frame_offset=None):
    """ Save video clip - either SLEAP annotated video or direct from camera frames
        Input:
            root - video output root
            filename - video output filename
            stims - stimulus onset frames, from relevant_session_frames
            ends - trial end frames, from relevant_session_frames
            labels_filename - filename for labels file (if using SLEAP  video)
            flags to choose which camera, fps of saved video, what the colour video frame offset
        Output: 
            video file saved to chosen path (either from SLEAP output or directly from camera video) """
    
    
    stim_start_frame = stims.iloc[trial]
    trial_end_frame = ends.iloc[trial]

    # account for color camera offset if specified
    if color_frame_offset:
        stim_start_frame = stim_start_frame - color_frame_offset
        trial_end_frame = trial_end_frame - color_frame_offset    
    
    # if SLEAP video
    if labels_filename:

        output = f'{outputPath}{os.sep}part{stim_start_frame}-{trial_end_frame}.avi'
        # extract LabeledFrames
        labels = sleap.load_file(labels_filename)
        video = labels.video
        labeledFrames = labels.labeled_frames

        # extract trial frame range
        labels_subset = labels.find(video, range(stim_start_frame, trial_end_frame+1))
        # make video from range
        sleap.io.visuals.save_labeled_video(output, labels, video, \
            frames=list(range(stim_start_frame, trial_end_frame+1)), fps=50, marker_size=2, show_edges=True)

        convert_avi_to_mp4(f'{outputPath}{os.sep}test.avi', f'{outputPath}{os.sep}test')
    
    # if not SLEAP video
    else:

        video_path = root + os.sep + filename
        start, end = stim_start_frame, trial_end_frame
        output = f"{outputPath}{os.sep}part{start}-{end}.avi"
        
        # load colorcamera video and define clip 
        cap = cv2.VideoCapture(video_path)
        # begin reading at start of clip to save time
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - 100)
        parts = [(start, end)]
        ret, frame = cap.read()
        h, w, _ = frame.shape

        if fps:
            framerate = fps
        elif colorVideo:
            framerate = 60.0
        else:
            framerate = 50.0

        # define writers for writing video clip
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writers = [cv2.VideoWriter(output, fourcc, framerate, (w, h)) for start, end in parts]

        # for each clip, loop through all video frames and only write the ones within clip indexes
        f = start - 100
        while ret:
            f += 1
            for i, part in enumerate(parts):
                start, end = part
                if start <= f <= end:
                    writers[i].write(frame)
            ret, frame = cap.read()

            # end reading after clip to save time
            if f > end + 100:
                break

        # release writers and capture
        for writer in writers:
            writer.release()

        cap.release()

    print("Video saved successfully")
    return output

# UNUSED
def save_color_video(root, fileName, outputPath, stim_startFrame, trial_endFrame, labels_fileName=None, colorVideo=False):
    """ Save color video clip """
    
    video_path = root + os.sep + fileName
    start, end = stim_startFrame, trial_endFrame
    output = f"{outputPath}{os.sep}part{start}-{end}.avi"
    if colorVideo:
        framerate = 60
    else:
        framerate = 50

    
    # load colorcamera video and define clip 
    cap = cv2.VideoCapture(video_path)
    # begin reading at start of clip to save time
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 100)
    parts = [(start, end)]
    ret, frame = cap.read()
    h, w, _ = frame.shape

    # define writers for writing video clip
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writers = [cv2.VideoWriter(output, fourcc, framerate, (w, h)) for start, end in parts]

    # for each clip, loop through all video frames and only write the ones within clip indexes
    f = start - 100
    while ret:
        f += 1
        for i, part in enumerate(parts):
            start, end = part
            if start <= f <= end:
                writers[i].write(frame)
        ret, frame = cap.read()

        # end reading after clip to save time
        if f > end + 100:
            break

    # release writers and capture
    for writer in writers:
        writer.release()

    cap.release()

    print("Video saved successfully")
    return output

# utility function for video conversion
def convert_avi_to_mp4(avi_file_path, output_name):
    """ convert .avi to .mp4 to allow embedding in jupyter """
    
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

# utility function to play saved video
def play_video(videoPath, colorVideo=False):
    """ Playback video clip (reuse code here)
        Call this as optional in save_video """

    # play trial video with opencv
    cap = cv2.VideoCapture(videoPath)

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

    return None
        
## TESTING ONLY
# load (download if refreshFiles) octpy metadata and video metadata for a specified session
# for each trial, extract event frames and save/play videos for specified trials
# currently using this data to feed into head_angle.py
if __name__ == '__main__':
    video_root = '/home/tomhagley/Documents/SLEAPProject/octagon_solo'
    video_filename = 'CameraTop_2022-11-02T14-00-00.avi'
    video_path = video_root + os.sep + video_filename

    directory_sleap = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
    fileName_sleap = 'model5_predictions_230206_CLI._CameraTop_2022-11-02T14-00-00.analysis.h5'
    sleap_path = directory_sleap + os.sep + fileName_sleap

    save_video_output_path = f'/home/tomhagley/Documents/SLEAPProject/octagon_solo/analysis'
    data_root = '/home/tomhagley/Documents/SLEAPProject/data'
    data_type = 'aeon_mecha'
    session = '2022-11-02_A006'

    trial = 91

    # load all data
    octpyMetadata, videoMetadata_list, colorVideoMetadata_list = access_metadata(root=data_root, session=session, colorVideo=True, refreshFiles=True)
    print(len(octpyMetadata), len(videoMetadata_list), len(colorVideoMetadata_list))

    # identify relevant frames from session: 1 trial (see code snippets)
    greyTrial, greyStim, greyEnd, colorTrial, colorStim, colorEnd = relevant_trial_frames(octpyMetadata, videoMetadata_list, colorVideoMetadata_list, trial)
    print(greyTrial, greyStim, greyEnd, colorTrial, colorStim, colorEnd)

    # identify relevant frames from session: all trials
    grey_trials, grey_stims, grey_ends, color_trials, color_stims, color_ends \
        = relevant_session_frames(octpyMetadata, videoMetadata_list, colorVideoMetadata_list)

    # identify relevant frames from filtered session: all trials
    octpy_metadata_choice = data_filtering.choice_trials_only(octpy_metadata=octpyMetadata)
    grey_trial_choice, grey_stim_choice, grey_end_choice, \
    color_trial_choice, color_stim_choice, color_end_choice = relevant_session_frames(octpy_metadata_choice, videoMetadata_list, \
                                                                                                          colorVideoMetadata_list)
                                                                                                          
                                                                                                                    
    # save video - greyscale
    video_filename = 'CameraTop_2022-11-02_all.avi'
    outputPath_greyscale = save_video(video_root, video_filename, save_video_output_path, grey_stims, grey_ends)
    play_video(outputPath_greyscale)

    # save video - color
    video_filename = 'CameraColorTop_2022-11-02T14-00-00.avi'
    outputPath_color = save_video(video_root, video_filename, save_video_output_path, color_stims, color_ends, colorVideo=True, fps=35, color_frame_offset = 52)
    play_video(outputPath_color)

    # return timestamps of every frame within each trial (stim on to trial end) for a session
    timestamp_list = timestamps_within_trial(grey_stims, grey_ends, videoMetadata_list)


    ### Video timestamp bug? ###
#     # save video - greyscale
#     outputPath = save_video(video_root, video_filename, save_video_output_path, 0, 100)
#     play_video(outputPath)

#     # save video - color
#     video_filename = 'CameraColorTop_2022-11-02T14-00-00.avi'
#     outputPath = save_video(video_root, video_filename, save_video_output_path, 52, 172)
#     play_video(outputPath)