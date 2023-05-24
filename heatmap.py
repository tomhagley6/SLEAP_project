from session import Session
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

""" Script to try out heatmap code """

root = '/home/tomhagley/Documents/SLEAPProject'
session = '2022-11-04_A004'
sess = Session(root, session)
sess.data_filtering('all')
sess.extract_basic_data()

# get the distances to wall 1 and 2 at the start frame of the trial
distances_wall_1 = sess.distances_wall1
distances_wall_2 = sess.distances_wall2
# also choice
chose_high = sess.octpy_metadata[sess.octpy_metadata['chose_light'] == False]
chose_high = sess.octpy_metadata.chose_light
chose_high = np.where(chose_high==True, False, True).astype('int')

# bin distances_wall_1 into 10
nbins = 4
bins = np.linspace(0, 2, nbins + 1)
binned_dist_wall_1 = np.digitize(distances_wall_1, bins) - 1
binned_dist_wall_2 = np.digitize(distances_wall_2, bins) - 1

# # initialise array
# arr = np.ones((chose_high.shape[0], chose_high.shape[0]))
# arr[:] = list()

# create nested list
lists_list = [[] for i in range((len(bins) - 1) ** 2)]

# append each trial to relevant list
for trial in range(binned_dist_wall_1.size):
    dist_wall_1_bin = binned_dist_wall_1[trial]
    dist_wall_2_bin = binned_dist_wall_2[trial]
    list_idx = (dist_wall_1_bin)*nbins + dist_wall_2_bin
    
    lists_list[list_idx].append(chose_high[trial])

for i in range(len(lists_list)):
    if bool(lists_list[i]):
        lists_list[i] = sum(lists_list[i]) / len(lists_list[i])
    else:
        lists_list[i] = np.nan

# convert to array and reshape to square matrix
arr = np.array(lists_list).reshape(nbins, nbins)

# create heatmap with a mask to ignore nan squares
p1 = sns.heatmap(arr, cmap='crest', mask=np.isnan(arr))
p1.set_xlabel('Distance from Low')
p1.set_ylabel('Distance from High')
plt.show()

def heatmap(x, y, val, nbins, cmap='crest', x_title=None, y_title=None):
    """ Function to plot a heat map given 1D arrays of two independent
        variables and the choice outcome.
        Function assumes 1 for choose_high, 0 for choose_low, so data
        must already be filtered to remove missed trials, forced-choice, etc.
         """


    # bin IVs 
    bins = np.linspace(0, 2, nbins + 1)
    binned_x = np.digitize(x, bins) - 1
    binned_y = np.digitize(y, bins) - 1

    # create nested list
    lists_list = [[] for i in range(nbins ** 2)]

    # append each trial to relevant list
    for trial in range(binned_x.size):
        x_bin = binned_x[trial]
        y_bin = binned_y[trial]
        list_idx = (y_bin)*nbins + x_bin
        
        lists_list[list_idx].append(val[trial])

    # collapse lists into mean average, or np.nan if empty
    for i in range(len(lists_list)):
        if bool(lists_list[i]):
            lists_list[i] = sum(lists_list[i]) / len(lists_list[i])
        else:
            lists_list[i] = np.nan

    # convert to array and reshape to square matrix
    arr = np.array(lists_list).reshape(nbins, nbins)

    # create heatmap with a mask to ignore nan squares
    p1 = sns.heatmap(arr, cmap='crest', mask=np.isnan(arr))
    p1.set_xlabel(x_title)
    p1.set_ylabel(y_title)
    plt.show()