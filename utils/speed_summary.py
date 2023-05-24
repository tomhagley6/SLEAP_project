import math
import numpy as np

GREYSCALE_FRAMERATE = 50

def find_speed_mean_STD(filtered_speed):
    """ Return the mean and std of a filtered trial speed profile,
        cropped to remove the first and last 0.5 seconds (irrelevant for
        change_of_mind and usually drops speed to 0) """

    # params
    greyscale_framerate = GREYSCALE_FRAMERATE
    # ignore first and last 0.5 seconds
    half_second = int(greyscale_framerate*0.5)
    
    cropped_speed = filtered_speed[half_second:-half_second]

    if cropped_speed.size == 0:
        return np.nan, np.nan

    # find average speed
    mean_speed = np.mean(cropped_speed)

    # find std
    speed_std = np.std(cropped_speed)

    return mean_speed, speed_std


def find_cropped_indexes(filtered_speed):
    """ find the index values of the trial included in the cropped
     version (for plotting) """

     # params
    greyscale_framerate = GREYSCALE_FRAMERATE

    half_second = int(greyscale_framerate*0.5)

    cropped_speed = filtered_speed[half_second:-half_second]

    if cropped_speed.size == 0:
        return np.nan, np.nan
    
    filtered_speed_length = filtered_speed.size

    return half_second, filtered_speed_length - half_second

def summary_stats_all_trials(filtered_speeds):
    """ find summary statistics of all concatenated trial
     speeds in a session (again cropped for first and last
     0.5s) """
    
    # concatenate all trial speeds in a session
    # crop trial speeds to remove the first and last 0.5 seconds
    cropped_filtered_speeds = []
    for trial in range(len(filtered_speeds)):
        cropped_x_start, cropped_x_end = find_cropped_indexes(filtered_speeds[trial])
        
        # don't include trials that last less than 1 s
        if not np.isnan(cropped_x_start):
            cropped_filtered_speeds.append(filtered_speeds[trial][cropped_x_start: cropped_x_end])

    # concatenate all trials together for summary stat purposes
    all_trial_speeds_concat = np.concatenate(cropped_filtered_speeds, axis=0)
    all_trial_speed_mean = np.mean(all_trial_speeds_concat)
    all_trial_speed_std = np.std(all_trial_speeds_concat)
    all_trial_median  = np.median(all_trial_speeds_concat)
    all_trial_IQR = np.subtract(*np.percentile(all_trial_speeds_concat, [75, 25]))

    return all_trial_speeds_concat, all_trial_speed_mean, all_trial_speed_std