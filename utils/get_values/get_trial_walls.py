def get_trial_walls(trial_type, walls):
    """ returns a list of ints: [light_wall, grating_wall]
    
        As of 230407, octpy stores walls as "light wall, grating wall"
        Reverse that order for the list here, so grating is always first
        For any function that gets these trial walls, reverse the list again
        if the light wall number should be in index 0
        """
    # current_trial = octpy_metadata.iloc[trial]
    # walls = current_trial.wall
    # trial_type = current_trial.trial_type

    intWalls = [int(x) for x in walls.split(',')]
    if trial_type == 'choiceLG' or trial_type == 'choiceGL':
        intWalls.reverse()

    return intWalls

def get_trial_walls_no_flip(octpy_metadata, trial):
    """ returns a list of ints: [CCW_wall, CW_wall] """
    current_trial = octpy_metadata.iloc[trial]
    walls = current_trial.wall

    intWalls = [int(x) for x in walls.split(',')]

    return intWalls


def get_trial_walls_session(trial_types, walls):
    """ repeat get_trial_walls over session data """

    int_walls_list = []
    for trial in range(len(trial_types)):
        pass
    return None