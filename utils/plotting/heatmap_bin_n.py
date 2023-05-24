import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
import math

# minimum data size to be included as a bin
MIN_N = 6

def heatmap_bin_n(x, y, val, nbins, bin_max, bin_min=0, v_max=1, v_min=0, cmap='crest', x_title=None, y_title=None, cut_low_n=False):
    """ Function to plot a heat map given 1D arrays of two independent
        variables and the choice outcome (or other dependent var).
        Function assumes 1 for choose_high, 0 for choose_low if using choice as dependent var, so data
        must already be filtered to remove missed trials, forced-choice, etc.

        Inputs:
        x - data to be binned along x axis
        y - data to be binned along y axis
        val - dependent variable (will be averaged across all data in the same bin)
        bin_max - max value of dependent var range to bin over
        bin_min - min value
        v_max - max value of val (1 for choice)
        v_min - min value

        Outputs:
        None. Draw heatmap
         """
    # 2.01

    # bin IVs 
    bins = np.linspace(bin_min, bin_max, nbins + 1)
    binned_x = np.digitize(x, bins) - 1
    binned_y = np.digitize(y, bins) - 1

    # create nested list
    lists_list = [[] for i in range(nbins ** 2)]

    # append each trial to relevant list
    for trial in range(binned_x.size):
        x_bin = binned_x[trial]
        y_bin = binned_y[trial]
        list_idx = (y_bin)*nbins + x_bin
        
        try: 
            lists_list[list_idx].append(val[trial])

        # catch cases where values are over/under maximum bin sizes, and discard them
        except IndexError:
            warnings.warn(f"for trial {trial}, bin index was out of range. Skipping trial.")
            
            continue
    
    # find matrix of bin sizes
    n_matrix = np.array([len(x) for x in lists_list]).reshape(nbins,nbins)
    print(np.flip(n_matrix, axis=0))

    # collapse lists into mean average, or np.nan if empty
    for i in range(len(lists_list)):
        if bool(lists_list[i]):
            # lists_list[i] = sum(lists_list[i]) / len(lists_list[i])
            lists_list[i] = np.nanmean(lists_list[i]) # use nanmean to avoid nan errors
        else:
            lists_list[i] = np.nan

    # convert to array and reshape to square matrix
    heatmap_bin_means = np.array(lists_list).reshape(nbins, nbins)

    # remove bin combinations with less than 10 trials
    if cut_low_n:
        heatmap_bin_means[n_matrix < MIN_N] = np.nan

    # find bin-centres to plot as x/y_ticks
    bin_centres = []
    for i in range(nbins):
        bin_centre = (bins[i] + bins[i+1])/2
        # change d.p. depending on type of data
        if int(bin_max) == 180:
            bin_centres.append(f"{bin_centre:.0f}")
        else:
            bin_centres.append(f"{bin_centre:.1f}")



    # create heatmap with a mask to ignore nan squares
    sns.set(font_scale=1.2)
    fig, p1 = plt.subplots()
    p1 = sns.heatmap(n_matrix, cmap='rocket', 
                     xticklabels=bin_centres, yticklabels=bin_centres,
                     annot=True,
                     mask=np.isnan(heatmap_bin_means))
    p1.set_facecolor('none')
    p1.set_xlabel(x_title, fontsize=15)
    p1.set_ylabel(y_title, fontsize=15)
    p1.invert_yaxis()
    plt.axis('scaled')
    plt.subplots_adjust(bottom=0.15)


    plt.show()