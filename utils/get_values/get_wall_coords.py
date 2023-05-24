import numpy as np

def get_wall_coords():
    """ getter for wall coords
        Output:
        wall_x_coords: list
        wall_y_coords: list """
    
    # starting from rightmost wall and going clockwise, in pixel coordinates
    wall_x_coords = [1177, 1041, 698, 360, 220, 363, 700, 1036]
    wall_y_coords = [547, 205, 63, 208, 549, 885, 1023, 884]

    # subtract y coordinates from ymax to find coords in image convention
    wall_y_coords = np.array(wall_y_coords)
    wall_y_coords = 1080 - wall_y_coords
    wall_y_coords = list(wall_y_coords)

    return wall_x_coords, wall_y_coords