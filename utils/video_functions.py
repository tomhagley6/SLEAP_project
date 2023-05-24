import numpy as np
import math
import warnings
import cv2
import sleap
import os
from sleap.io.visuals import img_to_cv, save_labeled_video

FRAMERATE_GREY = 50
FRAMERATE_COLOR = 20

def save_video_trial(video_root, stims, ends, trial, labels_file=None, video_filename=None, output_root=None, colorVideo=False, fps=False, color_frame_offset=None):
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

    # params
    stim_start_frame = stims.iloc[trial]
    trial_end_frame = ends.iloc[trial+2]  #### CHANGE THIS BACK
    framerate_grey = FRAMERATE_GREY
    framerate_color = FRAMERATE_COLOR

    # find output path
    if output_root:
        file_output_path = output_root
    else:
        file_output_path = video_root

    # account for color camera offset if specified
    if color_frame_offset:
        stim_start_frame = stim_start_frame - color_frame_offset
        trial_end_frame = trial_end_frame - color_frame_offset    
    
    # if SLEAP video
    if labels_file:
        print("This isn't happening")

        output = f'{file_output_path}{os.sep}part{stim_start_frame}-{trial_end_frame}.avi'
        # extract LabeledFrames
        labels = labels_file
        video = labels.video
        labeledFrames = labels.labeled_frames

        # extract trial frame range
        labels_subset = labels.find(video, range(stim_start_frame, trial_end_frame+1))
        # make video from range
        sleap.io.visuals.save_labeled_video(output, labels, video, \
            frames=list(range(stim_start_frame, trial_end_frame+1)), fps=framerate_grey, marker_size=2, show_edges=True)

        convert_avi_to_mp4(f'{file_output_path}{os.sep}part{stim_start_frame}-{trial_end_frame}_trial_{trial}.avi', 
                           f'{file_output_path}{os.sep}part{stim_start_frame}-{trial_end_frame}_trial_{trial}')
    
    # if not SLEAP video
    elif video_filename:

        video_path = video_root + os.sep + video_filename
        start, end = stim_start_frame, trial_end_frame
        output = f"{file_output_path}{os.sep}part{start}-{end}_trial_{trial}.avi"
        
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
            framerate = framerate_color
        else:
            framerate = framerate_grey

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

    else:
        warnings.warn("No labels file or video file given")

        return None

    print("Video saved successfully")

    return output

def convert_avi_to_mp4(avi_file_path, output_name):
    """ convert .avi to .mp4 to allow embedding in jupyter """
    
    os.popen("ffmpeg -y -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

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
        