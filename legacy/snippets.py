### head_angle.py ###
 ## head_angle_to_wall ##

# separate high/low
# only use choice trials
wall_high = []
wall_low = []
for wall in walls:
    if trialType == 'choiceLG':
        wall_low.append(wall[0])
        wall_high.append(wall[1])
    elif trialType == 'choiceGL':
        wall_high.append(wall[0])
        wall_low.append(wall[1])
    else:
        continue

# index the angles of the two walls
wall_high_ang, wall_low_ang = []
for i in range(len(wall_high)):
    wall_high_ang.append(wall_angles[wall_high[i]])
    wall_low_ang.append(wall_angles[wall_low[i]])

# find head angle relative to walls
head_angle = head_angle
head_angle_wall_high = math.abs(head_angle - wall_high_ang[0])
head_angle_wall_low = math.abs(head_angle - wall_low_ang[1])

# return whether closest to low (in angle)
ang_close_to_low = False
if head_angle_wall_low < head_angle_wall_high:
    ang_close_to_low = True

### find_frames.py ###
 ## relevant_frames_from_session_vectorised ##
 # when this was not vectorised #

def relevant_frames_from_session(octpyMetadata, videoMetadata_list, colorVideoMetadata_list, trial):
    """ Return trial_start, stim_start, and response frames for each trial in a session """
    
    # TODO: vectorise
    trial = trial

    # only working for one video - metadata list len 1
    videoMetadata = videoMetadata_list[0]
    colorVideoMetadata = colorVideoMetadata_list[0]

    # relevant trial frames
    # params
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
    trialStart_idx = videoMetadata.index.get_loc(trialStart_time, method='ffill')
    trialStart_frame = videoMetadata.iloc[trialStart_idx]
    trialStart_frameNum = trialStart_frame._frame

    # trial_end
    # Remember RT is from trial_onset, not stim_onset
    # Convert RT to num frames after stimStart_time
    naiveRT = octpyMetadata.iloc[trial].RT
    RT = pd.Timedelta(naiveRT, 's') - (stimStart_time - trialStart_time)
    RT = RT/pd.Timedelta(1,'s') # convert from Timedelta to seconds float
    RT_frames = math.ceil(RT*framerate_greyscale)
    trialEnd_frameNum = stimStart_frameNum + RT_frames


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
    stimStart_idx_color = colorVideoMetadata.index.get_loc(stimStart_time, method='ffill')
    stimStart_frame_color = colorVideoMetadata.iloc[stimStart_idx_color]
    stimStart_frameNum_color = stimStart_frame_color._frame
    # trial start
    trialStart_idx_color = colorVideoMetadata.index.get_loc(trialStart_time, method='ffill')
    trialStart_time_color = colorVideoMetadata.index.values[trialStart_idx_color]
    trialStart_frame_color = colorVideoMetadata.iloc[trialStart_idx_color]
    trialStart_frameNum_color = trialStart_frame_color._frame
    # trial end 
    RT_frames = math.ceil(RT*framerate_color)
    trialEnd_frameNum_color = stimStart_frameNum_color + RT_frames

    return trialStart_frameNum, stimStart_frameNum, trialEnd_frameNum, trialStart_frameNum_color, stimStart_frameNum_color, trialEnd_frameNum_color

### utils/normalising.py ###
# replaced by single function for x or y

def normalise_y(y):
# normalise distance using centre and top-left coordinates
    centre = 536
    top_left = 66
    point = y
    half_arena = centre - top_left

    diff = point - top_left
    normalised_diff = diff/(2*half_arena)

    if normalised_diff > 1:
        normalised_diff = 1
    elif normalised_diff < 0:
        normalised_diff = 0

    # convert to [-1,1] range
    normalised_diff = normalised_diff*2 - 1

    return normalised_diff