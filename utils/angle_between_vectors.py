import math
import numpy as np


def angle_between_vectors(coords):
    """ return angle between 2 vectors """
    a,b,c,d = coords
    v1, v2 = np.array((a,b)), np.array((c,d))

    dotProduct = v1[0]*v2[0] + v1[1]*v2[1]
    v1Magnitude_sq = v1[0]**2 + v1[1]**2
    v2Magnitude_sq = v2[0]**2 + v2[1]**2
    v1Magnitude = math.sqrt(v1Magnitude_sq)
    v2Magnitude = math.sqrt(v2Magnitude_sq)

    cosTheta = dotProduct/(v1Magnitude*v2Magnitude)
    theta = math.acos(cosTheta)
    
    return theta