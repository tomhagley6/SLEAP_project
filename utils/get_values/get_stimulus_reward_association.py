from utils.manipulate_data.data_filtering import choice_trials_only, sub_x_RT_only, filter_miss_trials, chose_light_only

def grating_is_high(octpy_metadata):
    """ from octpy metadata, get whether this session
     is a light == High or grating == High session
     
     Return True if grating == High, return False if 
     light == High"""
    
    # filter om to remove forced trials, RT over 10 s, 
    # miss trials, and choose_grating trails
    # om = choice_trials_only(octpy_metadata)
    # om = chose_light_only(om)
    # om = filter_miss_trials(om)
    # om = sub_x_RT_only(om, 10)

    om = octpy_metadata[
                        (octpy_metadata['choice_trial'] == True)
                        & (octpy_metadata['chose_light'] == True)
                        & (octpy_metadata['miss_trial'] == False)
                        & (octpy_metadata['RT'] < 10)
                        ]
    

    # take the first index of the remaining trials,
    # identify what the reward was for choosing light
    # use this to identify session paradigm
    trial = om.iloc[0]
    if trial.reward <= 1: # light was low
        return True
    elif trial.reward > 1: # light was high
        return False
    else:
        print("Error with reward.")
        return None
