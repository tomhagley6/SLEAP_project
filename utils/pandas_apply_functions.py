
""" collection of pandas apply functions intended for octpy_metadata """

# pd apply function
# Assumes grating is high!
def order_walls(walls, trial_type):
    """ reverse order of walls in OctPy if trial_type is LG """
    intWalls = [int(x) for x in walls.split(',')]
    if trial_type == 'choiceLG':
        intWalls.reverse()
    
    return intWalls