from utils.plotting.trial_start_test_plot import test_head_angle_walls_distance
from session import Session


root = '/home/tomhagley/Documents/SLEAPProject'
session = '2022-11-04_A004'
sess = Session(root, session)
sess.extract_basic_data()

for i in range(0,1):
    test_head_angle_walls_distance(sess.octpy_metadata, i, sess.grey_stims,
                                                        sess.labels_file, 0, plotFlag=True)    