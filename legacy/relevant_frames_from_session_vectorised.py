### find_frames.py ###
 ## relevant_frames_from_session_vectorised ##
 # when this function was a work in progress and included both #

# double check that stim_on time is accurate from bonsai hardware
def relevant_frames_from_session_vectorised(octpyMetadata, videoMetadata_list, colorVideoMetadata_list, trial):
    """ Return trial_start, stim_start, and response frames for each trial in a session """
    
    # TODO: vectorise
    trial = trial

    # only working for one video - metadata list len 1
    vm = videoMetadata_list[0]
    cvm = colorVideoMetadata_list[0]

    om = octpyMetadata

    # relevant trial frames
    # params
    framerate_greyscale = 50
    # trial_start
    trialStart_time = om.index.values[trial]
    # stim_start
    stimDelay = om.iloc[trial].stimulus_delay
    stimDelay_frames = stimDelay*framerate_greyscale
    stimStart_time = trialStart_time + pd.Timedelta(stimDelay, 's') # need to use pandas timedelta if passing a float 

    ## vectorised
    # trial_start
    trialStart_times = pd.Series(om.index.values)
    # stim_start
    om['stimStart_time'] = om['ts'] + om['stimulus_delay'].apply(pd.to_timedelta, unit='S')
    stimStart_times = om.reset_index().stimStart_time # replace trial_start_time index with ordinal index

    # frame index for stim_start
    stimStart_idx = vm.index.get_loc(stimStart_time, method='ffill')
    stimStart_frame = vm.iloc[stimStart_idx]
    stimStart_frameNum = stimStart_frame._frame
    trialStart_idx = vm.index.get_loc(trialStart_time, method='ffill')
    trialStart_frame = vm.iloc[trialStart_idx]
    trialStart_frameNum = trialStart_frame._frame
    
    ## vectorised
    # use merge_asof instead of index.get_loc for vectorised operation (must be sorted by the key column)
    # returns a left-join merge of the two dataframes, but can extract the column that gives us vm camera frame times
    # left_on and right_on are redundant as we are passing single-column df
    stimStart_vmTimes = pd.merge_asof(om.stimStart_time, vm.reset_index(level=0).time, left_on='stimStart_time', right_on='time', direction='backward')
    stimStart_vmTimes = stimStart_vmTimes.time
    stimStart_vm = vm.loc[list(stimStart_vmTimes)]
    stimStart_frameNums = stimStart_vm._frame

    trialStart_vmTimes = pd.merge_asof(om.ts, vm.reset_index(level=0).time, left_on='ts', right_on='time', direction='backward')
    trialStart_vmTimes = trialStart_vmTimes.time
    trialStart_vm = vm.loc[list(trialStart_vmTimes)]
    trialStart_frameNums = trialStart_vm._frame


    # trial_end
    # Remember RT is from trial_onset, not stim_onset
    # Convert RT to num frames after stimStart_time
    naiveRT = octpyMetadata.iloc[trial].RT
    RT = pd.Timedelta(naiveRT, 's') - (stimStart_time - trialStart_time)
    RT = RT/pd.Timedelta(1,'s') # convert from Timedelta to seconds float
    RT_frames = math.ceil(RT*framerate_greyscale)
    trialEnd_frameNum = stimStart_frameNum + RT_frames


    ## vectorised
    naiveRTx = om.reset_index().RT
    RTx = naiveRTx.apply(pd.to_timedelta, unit='S') - (stimStart_times - trialStart_times)
    RTx_float = RTx/pd.Timedelta(1,'s')
    RT_framesx = np.ceil(RTx_float*framerate_greyscale)
    trialEnd_frameNums_naive = stimStart_frameNums.reset_index()._frame + RT_framesx.astype('int64') # can only add series if they have compatible/no index
    trialEnd_frameNums_naive = trialEnd_frameNums_naive.rename('trialEnd_frameNums_naive')
    trialEnd_times = stimStart_times + RTx
    trialEnd_times = trialEnd_times.rename('trialEnd_times') # cannot merge a series without a name
    
    # match forward here because we want to be inclusive of trial end
    trialEnd_vmTimes = pd.merge_asof(trialEnd_times, vm.reset_index(level=0).time, left_on='trialEnd_times', right_on='time', direction='forward')
    trialEnd_vmTimes = trialEnd_vmTimes.time
    trialEnd_vm = vm.loc[list(trialEnd_vmTimes)]
    trialEnd_frameNums = trialEnd_vm._frame

    timeStamp = "2022-11-02 14:33:00"
    timeSlice = octpyMetadata.loc[: '2022-11-02 14:30:40']

    # find colorvideo framerate
    startIdx = 0
    startTimestamp = cvm.index.values[0]
    endTimestamp = startTimestamp + pd.Timedelta(1, 's')
    # framerate is idx of nearest timestamp to startTime + 1 s
    endIdx = cvm.index.get_loc(endTimestamp, method= 'nearest')
    framerate_color = endIdx

    # find stim_on frame in colorvideo
    stimStart_idx_color = cvm.index.get_loc(stimStart_time, method='ffill')
    stimStart_frame_color = cvm.iloc[stimStart_idx_color]
    stimStart_frameNum_color = stimStart_frame_color._frame
    trialStart_idx_color = cvm.index.get_loc(trialStart_time, method='ffill')
    trialStart_time_color = cvm.index.values[trialStart_idx_color]
    trialStart_frame_color = cvm.iloc[trialStart_idx_color]
    trialStart_frameNum_color = trialStart_frame_color._frame

    # vectorised
    # stim start
    stimStart_vmTimes_color = pd.merge_asof(om.stimStart_time, cvm.reset_index(level=0).time, left_on='stimStart_time', right_on='time', direction='backward')
    stimStart_vmTimes_color = stimStart_vmTimes_color.time
    stimStart_vm_color = cvm.loc[list(stimStart_vmTimes_color)]
    stimStart_frameNums_color = stimStart_vm_color._frame

    #trial state
    trialStart_vmTimes_color = pd.merge_asof(om.ts, vm.reset_index(level=0).time, left_on='ts', right_on='time', direction='backward')
    trialStart_vmTimes_color = trialStart_vmTimes.time
    trialStart_vm_color = vm.loc[list(trialStart_vmTimes)]
    trialStart_frameNums_color = trialStart_vm._frame

    # trial end
    RT_framesx_color = np.ceil(RTx_float*framerate_color)
    trialEnd_frameNums_color_naive = stimStart_frameNums + RT_framesx_color
    
    # match forward here because we want to be inclusive of trial end
    trialEnd_vmTimes_color = pd.merge_asof(trialEnd_times, cvm.reset_index(level=0).time, left_on='trialEnd_times', right_on='time', direction='forward')
    trialEnd_vmTimes_color = trialEnd_vmTimes_color.time
    trialEnd_vm_color = cvm.loc[list(trialEnd_vmTimes_color)]
    trialEnd_frameNums_color = trialEnd_vm_color._frame

    return trialStart_frameNums, stimStart_frameNums, trialEnd_frameNums, trialStart_frameNums_color, stimStart_frameNums_color, trialEnd_frameNums_color