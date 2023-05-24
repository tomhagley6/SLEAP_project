import os
import warnings
import sleap
import find_frames
import utils.manipulate_data.data_filtering as data_filtering
import trajectory_extraction
import head_angle
import distances
import head_angle_analysis
from utils.trajectory_speeds import trajectory_speeds
from utils.change_of_mind import change_of_mind_session, save_CoM_videos
from utils.h5_file_extraction import get_node_names, get_locations
from utils.correct_color_camera_frame_offset import find_color_camera_frame_offset
from utils.get_values.get_wall_numbers import get_wall1_wall2
from utils.node_coordinates import node_coordinates_at_frame_session
from utils.head_angle_to_wall import head_angle_to_wall_session
from utils.get_values.get_head_to_wall_angles import get_head_to_wall_angle_trial_starts_NEW, get_head_to_wall_angle_full_trial_full_session
from utils.distance_to_wall import distance_to_wall_trial_start_sess
from utils.get_values.get_real_RT import response_times_session
from time_to_alignment import time_to_alignment_session
from utils.get_values.get_stimulus_reward_association import grating_is_high
from analysis.logistic_regression import logistic_regression_choose_high
import numpy as np
import matplotlib.pyplot as plt



class Session():
    """ define class for a single session and its extracted data
        Instantiate with top-level SLEAP project directory, session tag, and social flag """
    
    def __init__(self, root=None, session=None, social=False) -> None:
        """ initialise Session class with all paths """
        self.root = root
        self.session = session
        self.social = social
        self.project_type = 'octagon_multi' if social else 'octagon_solo'
        self.data_path = self.root + os.sep + 'data'
        self.video_path = self.root + os.sep + f'{self.project_type}' + os.sep + 'videos' + os.sep + f'{session}'
        self.trajectories_path = self.root + os.sep +  f'{self.project_type}' + os.sep + 'exports' + os.sep + f'{session}' + os.sep + f'CameraTop_{session}' + '_analysis.h5'
        self.labels_path = (self.root + os.sep + f'{self.project_type}/predictions/{session}' + 
                             os.sep +f'CameraTop_{session}' + '_predictions.slp')
        self.video_output_path = (self.root + os.sep + f'{self.project_type}/analysis')
        self.video_filename = f"CameraTop_{self.session}_all.avi"
        self.color_video_filename = f"CameraColorTop_{self.session}_all.avi"
        self.labels_file = sleap.load_file(self.labels_path)
        
        # load metadata files from HPC using octpy
        # if files are not already local, refresh data access from HPC
        try:
            self.octpy_metadata, self.video_metadata,\
                            self.color_video_metadata = find_frames.access_metadata(self.data_path,
                                                                                    session, 
                                                                                    colorVideo=True,
                                                                                    refreshFiles=False)
        except (FileNotFoundError, IndexError):
            self.octpy_metadata, self.video_metadata,\
                            self.color_video_metadata = find_frames.access_metadata(self.data_path, 
                                                                                    session, 
                                                                                    colorVideo=True, 
                                                                              refreshFiles=True)

        
    def data_filtering(self, filter_type):
        """ optionally filter octpy_metadata 
            Input: 
                filter_type - string to choose type of filter, or 'all' for all"""
        
        # check prerequisites for function
        try:
            self.octpy_metadata
        except NameError:
            warnings.warn("Octpy metadata not defined. Exiting function.")

            return None
        
        # params
        RT_cutoff = 15


        if filter_type == 'choice_trials':
            self.octpy_metadata = data_filtering.choice_trials_only(self.octpy_metadata)
        elif filter_type == 'hit_trials':
            self.octpy_metadata = data_filtering.filter_miss_trials(self.octpy_metadata)
        elif filter_type == 'RT':
            self.octpy_metadata = data_filtering.sub_x_RT_only(self.octpy_metadata, RT_cutoff)
        elif filter_type == 'all':
            self.octpy_metadata = self.octpy_metadata[
                                    (self.octpy_metadata['choice_trial'] == True)
                                    & (self.octpy_metadata['miss_trial'] == False)
                                    & (self.octpy_metadata['RT'] < RT_cutoff)
                                    ]

        return None

    
    def extract_basic_data(self, centre_node='bodyupper', color_frame_offset=None):
        """ do basic data extraction after getting metadata
            Currently includes:
             - trial start, stim start, and trial end frames for each video
             - winner mouse tracks
             - trial trajectories
             - head angles to horizontal (trial start)
             - wall 1 for each trial
             - wall 2 for each trial
             - head angles to wall 1 (trial start)
             - head angles to wall 2 (trial start)
             - distances to wall 1 (trial start)
             - distances to wall 2 (trial start)"""
        
        # check prerequisites for function
        try:
            self.octpy_metadata
        except NameError:
            warnings.warn("Octpy metadata not defined. Exiting function.")

            return None
        
        # first identify whether session is grating == High or light == High
        self.grating_is_high = grating_is_high(self.octpy_metadata)
        
        # find ColorCamera and GreyscaleCamera frame numbers for trial start, stim start, and trial end
        self.grey_trials, self.grey_stims, self.grey_ends, \
        self.color_trials, self.color_stims, self.color_ends = find_frames.relevant_session_frames(self.octpy_metadata, 
                                                                                    self.video_metadata, 
                                                                                  self.color_video_metadata)
        if not color_frame_offset:
            # find color camera frame offset to correct for photodiode bug on octagon_1 (this should always be 0 on octagon 2)
            # do this for either start or end (start = False)
            self.color_frame_offset = find_color_camera_frame_offset(self.video_metadata, 
                                                                self.color_video_metadata, 
                                                                start=True)
            
        # BODY
        ### winner mouse track ###
        if self.social:
            # TODO find which track contains the winner mouse for each trial and assign to self
            # create a tracks array that records track of winner for each trial
            pass
        else:
            self.tracks = np.zeros(self.octpy_metadata.shape[0]).astype('int')

        ### TRAJECTORIES ###
        ### trial trajectories ###
        # find smoothed and normalised (interpolated) trajectories for self for each trial in the session 
        # (rotated/flipped if needed)
        self.trajectories = trajectory_extraction.extract_session_trajectories(self.trajectories_path, 
                                                                               self.grey_stims,
                                                                        self.grey_ends, 
                                                                        self.octpy_metadata, 
                                                                        self.tracks, 
                                                                        normalise=True, 
                                                                        smooth=True,
                                                                        flip_rotate=False)


        ### HEAD ANGLES ###
        # find head angles to horizontal and to relevant walls at the stim_start of trial
        ### head angle to horizontal ### 
        self.head_angles = head_angle.extract_head_angle_session(self.grey_stims, 
                                                                 self.labels_file, 
                                                                 self.tracks)

        ### CODE EDITS: new version of get_head_to_wall_angle_trial_starts, and extracting video trajectories here

        #self.video_trajectories, _ = trajectory_extraction.extract_video_trajectory(self.trajectories_path, normalise=True, smooth=True)

        
        ### head angle to high and low wall ### 
        self.head_angles_wall_1, self.head_angles_wall_2 = get_head_to_wall_angle_trial_starts_NEW(self.octpy_metadata, 
                                                                                      self.trajectories,
                                                                                    self.grey_stims, self.tracks, 
                                                                                    self.labels_file,
                                                                                    self.grating_is_high)

        ### head angle to other mouse ###
        if self.social:
            # TODO find head_angle to other mouse for social
            pass


        ### DISTANCES ### 
        # distances to wall
        if self.social:
            # TODO find distances to each wall for self and other mouse at start of trial
            pass
        else:
            # TODO find distances to each wall for self
            self.wall_1_session, self.wall_2_session = get_wall1_wall2(self.octpy_metadata,
                                                                       self.grating_is_high)
            self.distances_wall_1 = distance_to_wall_trial_start_sess(self.trajectories, 
                                                                     centre_node, 
                                                                     self.wall_1_session, 
                                                                     self.tracks)
            
            self.distances_wall_2 = distance_to_wall_trial_start_sess(self.trajectories, 
                                                                     centre_node, 
                                                                     self.wall_2_session, 
                                                                     self.tracks)


        # find misc interesting trajectory stuff:



        #  2. find trajectory distances
        # TODO

        return None
    
    
    # more involved data extraction
    def more_complex_extraction(self):
        """ More involved extraction
            Current includes:
             - full-video continuous trajectory
             - full-video continuous head_angle_to_wall
             - trial speed profiles
             - time to align head to trial walls for each trial """
        
        # check prerequisites for function
        try:
            self.octpy_metadata, self.trajectories
        except NameError:
            warnings.warn("prerequisite variables not defined. Exiting function.")

            return None

        ### TRAJECTORIES ###
        ### video trajectory ###
        # find continuous trajectory across full video
        # video_trajectories as a numpy array of shape (frames, nodex/y_coordinates, tracks)
        self.video_trajectories, col_names = trajectory_extraction.extract_video_trajectory(self.trajectories_path,
                                                                                       normalise=True, 
                                                                                       smooth=True)
        ### SPEED ###
        # first find the timestamps within each trial
        # timestamps list is a list of numpy arrays, containing the timstamps for every frame in each trial
        self.timestamps_list = find_frames.timestamps_within_trial(self.grey_stims, self.grey_ends, self.video_metadata)
        # trial_speed_profiles have the same shap as timestamps list. Gaussian filtered instantaneous speed
        self.trial_speed_profiles = trajectory_speeds(self.trajectories, self.timestamps_list)
        

        #  4. time to align head angle with walls
        # first find the head_to_wall_ang for every frame in every trial in session
        self.head_wall_1_ang_all_frames, \
        self.head_wall_2_ang_all_frames = get_head_to_wall_angle_full_trial_full_session(self.octpy_metadata, 
                                                                                         self.video_trajectories, 
                                                                                    self.grey_stims, self.grey_ends, 
                                                                                    self.tracks, self.labels_file,
                                                                                    self.grating_is_high)
        
        # now find the time to align with either of the two walls (angle specificity is in function file)
        self.times_to_head_wall_alignment = time_to_alignment_session(self.head_wall_1_ang_all_frames, 
                                                                      self.head_wall_2_ang_all_frames)
        # find the trial response times to compare with
        self.response_times_sess = response_times_session(self.octpy_metadata)

        return None
    

    def change_of_mind(self):
        """ Find change-of-mind trials for this session
            This is currently based on there being a certain length of 
            consecutive time in the trial where head angle aligns with 
            each of the two walls.
            Also, speed must dip below threshold after first reach a
            baseline speed
             
            Gives access to the idxs of the CoM trials, filtered
            octpy_metadata, and also saves videos
            
            CoM trials need manual curation (possibly manually changing 
            parameters in CoM function)
             
            Could be improved by using different heuristics between when 
            subject is close to the wall and when it is far away"""
        
        # check prerequisites
        try:
                self.octpy_metadata, self.trajectories, self.video_trajectories
        except NameError:
            warnings.warn("Prerequisite variables not defined. Exiting function.")

            return None
        
        #  3. find change-of-mind trials
        # use continuous trajectory for the whole video
        # now run through change_of_mind function and return CoM trial index values 
        # use inverse to switch between CoM and non-CoM

        trajectories = self.trajectories
        video_trajectories = self.video_trajectories
        labels_file = self.labels_file
        octpy_metadata = self.octpy_metadata
        grey_stims = self.grey_stims
        grey_ends = self.grey_ends
        color_stims = self.color_stims
        color_ends = self.color_ends
        tracks = self.tracks
        timestamps_list = self.timestamps_list
        video_path = self.video_path
        video_filename = self.video_filename
        color_frame_offset = self.color_frame_offset


        self.CoM_trial_indexes = change_of_mind_session(trajectories, video_trajectories, 
                                                        labels_file, octpy_metadata,
                                                grey_stims, grey_ends, tracks, 
                                                timestamps_list, video_path, video_filename,
                                                color_frame_offset=color_frame_offset, 
                                                save_videos=False, inverse=False)

        # create new octpy_metadata with only these trials (or only without)
        self.octpy_metadata_CoM = data_filtering.change_of_mind_trials( trajectories, video_trajectories,  
                                                                       labels_file, octpy_metadata,
                                                 grey_stims, grey_ends, tracks, timestamps_list, 
                                                  video_path, video_filename,
                                                color_frame_offset=color_frame_offset, save_videos=False)

        self.octpy_metadata_non_CoM = data_filtering.non_change_of_mind_trials( trajectories, video_trajectories,
                                                                                 labels_file, octpy_metadata,
                                                 grey_stims, grey_ends, tracks, timestamps_list, 
                                                  video_path, video_filename,
                                                color_frame_offset=color_frame_offset, save_videos=False)

        # save color video clips for CoM trials
        save_CoM_videos( video_path, self.color_video_filename, color_stims, color_ends, self.CoM_trial_indexes,
                        color_frame_offset)
        
        return None
        
    
    def find_color_frame_offset(self, video_metadata, color_video_metadata, start=True):
        """ find color camera frame offset to correct for photodiode bug on octagon_1 
        (this should always be 0 on octagon 2)
        Do this for either start or end (start = False) """
        color_frame_offset = find_color_camera_frame_offset(video_metadata,
                                                             color_video_metadata,
                                                               start=start)
        
        return color_frame_offset
    

    def get_extraction_level(self):
        """ Return information about how much extracted data is in the Session
            object """
        
        try:
            self.video_trajectories
            print("Full data extraction.")
        except AttributeError:
            try:
                self.trajectories
                print("Initial data extraction.")
            except AttributeError:
                try:
                    self.octpy_metadata
                    print("Only trial metadata extracted.")
                except AttributeError:
                    print("No data.")
