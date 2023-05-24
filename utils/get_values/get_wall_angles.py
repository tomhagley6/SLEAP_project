import numpy as np
from utils.angle_between_vectors import angle_between_vectors
import matplotlib.pyplot as plt

def get_wall_angles(plot_flag=False):
    """ Return angles of the 8 walls from rightward horizontal """
    # horizontal vector
    centre= np.array((699, 532))
    rightMiddle = np.array((1440, 532))
    rightVector = rightMiddle - centre

    # starting from rightmost wall and going clockwise
    wall_ycoords = [547, 205, 63, 208, 549, 885, 1023, 884]
    wall_xcoords = [1177, 1041, 698, 360, 220, 363, 700, 1036]

    # conver from coordinats to a vector from centre
    wall_ycoords_vector = [y_coord - centre[1] for y_coord in wall_ycoords]
    wall_xcoords_vector = [x_coord - centre[0] for x_coord in wall_xcoords]
    wall_vectors = list(zip(wall_xcoords_vector, wall_ycoords_vector))

    # use the vector find angle from horizontal
    wall_angles = []
    for i in range(len(wall_ycoords)):
        difference  = angle_between_vectors(np.hstack([wall_vectors[i], rightVector]))
        wall_angles.append((difference))

    # angle between vectors is symmetric above and below the horizontal vector (formula gives minimum angle)
    # so subtract angle from 2pi if below horizontal vector
    for i in range(len(wall_angles)):
        if wall_vectors[i][1] < 0:  # if y negative
            wall_angles[i] = 2*np.pi - wall_angles[i]

    if plot_flag:
        # plot
        fig = plt.figure(figsize=(12,12))
        # specify polar axes by polar=True or projection='polar'
        ax2 = fig.add_subplot(111, projection='polar')
        labels =  [f"Wall {i}" for i in range(8)]
        bars = ax2.bar(wall_angles, [1]*(len(wall_angles)), width=np.pi/16, bottom=0.0, alpha=0.5)
        ax2.bar_label(bars, labels=labels, padding=3)
        #bar.set_alpha(0.5)
        plt.show()

    return wall_angles