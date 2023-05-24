import h5py
import numpy as np
import pandas as pd
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
import sleap
import cv2 as cv
from utils.manipulate_data.interpolate import interpolate

## SCRIPT FOR TESTING, NOT FOR USE


# paths 
directory = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/exports'
fileName = 'CameraTop_2022-11-02_Full_predictions_model5_230228_CLI.analysis.h5'
filePath = directory + os.sep + fileName

# bonsai data
directory_bonsai = '/home/tomhagley/Documents/SLEAPProject/data/bonsai_metadata'
fileName_bonsai = '2022-11-02_A006.csv'
filePath_bonsai = directory_bonsai + os.sep + fileName_bonsai

# extract Sleap HDF5 data by slicing all values in dict entries
# transpose to covert from column major order
with h5py.File(filePath, 'r') as f:
    dsetNames = list(f.keys())
    trackOccupancy = f['track_occupancy'][:].T
    locations = f['tracks'][:].T
    nodeNames = [name.decode() for name in f['node_names'][:]] # decode string from UTF-8 encoding

# plotting params
sns.set_theme('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]

### Summaries ###

print("===filename===")
print(fileName, end='\n\n')

print("===HDF5 datasets===")
print(f"{', '.join(dsetNames)}", end='\n\n')

print("===locations data shape===")
print(locations.shape, end='\n\n')
frameCount, nodeCount, _, instanceCount = locations.shape

print("===mistracking===")
unoccupiedTracksIdx = np.where(trackOccupancy == False)[1]
print(f"Number of dropout frames: {np.sum(trackOccupancy == False)}", end='\n\n')

print("===nodes===")
for i, node in enumerate(nodeNames):
    print(f"{i}: {node}")

print(trackOccupancy.shape)
missingInstances = locations[unoccupiedTracksIdx]
trackedLocations = locations[np.where(trackOccupancy != False)[1]]
trackedLocationsIdx = np.where(trackOccupancy != False)[0]

### Interpolation ### 

# Store initial shape
locShape = locations.shape
# Reshape to flatten dimensions after the first
# Can use np.reshape to flatten some dimensions, because using '-1'in place of one of the output
# dimensions will cause numpy to infer its size from the remaining dimensions
# Take a copy copy or locationsFlat will be a view of locations
locationsFlat = locations.copy().reshape(locShape[0], -1)

# interpolate separtely for all x-y coordinates for all nodes for all instances
for i in range(locationsFlat.shape[1]):
    data = locationsFlat[:,i]
    # remove NaNs
    # flatnonzero can be used to return indices in a flattened version of the array
    nonNaNidx = np.flatnonzero(~np.isnan(data))
    # build interpolant
    f = scipy.interpolate.interp1d(nonNaNidx, data[nonNaNidx], kind='linear', fill_value=np.nan, bounds_error=False)
    # replace the indices that are NaN with interpolated values
    NaNidx = np.flatnonzero(np.isnan(data))
    data[NaNidx] = f(NaNidx)

    # fill leading or trailing NaNs with nearest non-NaN value
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])

    locationsFlat[:,i] = data

locationsInterp = locationsFlat.reshape(locShape)

### Smoothing ###

# Use savitsky-golay filter: 1-D smoothing filter that estimates each point by fitting 
# a polynomial to a small window around each point, and use the polynomial to estimate the
# central point (then iterating for the whole signal)

# filter all nodes
# flatten first for easier use 
locShape = locationsInterp.shape
flattenedLocs = locationsInterp.copy().reshape(locShape[0], -1)
for i in range(flattenedLocs.shape[1]):
    data = flattenedLocs[:,i]
    # can add derivative=1 here to filter the data and also differentiate 
    # This may be the best option because the fitted polynomials are being used
    # to give the differential at each point
    # This might just give back velocity in one shot
    data_filtered = scipy.signal.savgol_filter(data, window_length=25, polyorder=3)
    flattenedLocs[:,i] = data_filtered
locationsFiltered = flattenedLocs.reshape(locShape)

### Smoothing Visualisation ### 

bodyUpperIdx = 4
bodyUpperLoc = locationsInterp[:,bodyUpperIdx,:,:]
bodyUpperLoc_unfiltered = locationsFiltered[:,4,:,:]
earLoc = locations[:,2,:,:]
plt.plot(bodyUpperLoc[0:200,0,0], 'y', label='BodyUpper')
plt.plot(bodyUpperLoc_unfiltered[0:200,0,0], 'g', label='BodyUpper Smoothed')

plt.legend()
plt.title("Upper body position across 200 frames")
plt.show()

### distribution of coordinates ### 

plt.figure(3)
# plot distribution of x and y coordinates for 1 node
noseLocs_x = locations[:,0,0,0]
noseLocs_y = locations[:,0,1,0]
print(noseLocs_x.shape)
plt.subplot(211)
plt.hist(noseLocs_x)
plt.subplot(212)
plt.hist(noseLocs_y)
plt.show()


### create colourbar ###

timestamps = np.arange(2700)
min_val, max_val = min(timestamps), max(timestamps)
cmap = mpl.cm.summer
norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

""" fig, ax = plt.subplots()
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=sorted(timestamps), orientation='horizontal')
cb.set_label('Time (start to finish)')
ax.tick_params(axis='x', rotation=90) """


### plotting x-y trajectory ###

# plotting params
sns.set_theme('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [7,7]

noseLoc = locations[:,0,:,:]
plt.figure(figsize=(7,7))
sc = plt.scatter(noseLoc[0:2700,0,0], noseLoc[0:2700,1,0], s=3, c=timestamps, cmap=cmap, norm=norm)
#plt.colorbar(sc)
plt.xlim((100, 1300))
plt.ylim((-1150, 50))
plt.axis('scaled')
plt.show()

### plotting on video frames ###

video = sleap.load_video('/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraTop_2022-11-02T14-00-00.avi')
print(video.shape)
img = video[0]
img2 = img.reshape(1080, 1440)
img3 = img2[::-1,:]
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.scatter(600, 550)

plt.show()

noseLoc = locations[:,0,:,:]
plt.figure(2, figsize=(7,7))
plt.imshow(img3, cmap='gray', vmin=0, vmax=255)
sc = plt.scatter(noseLoc[0:2700,0,0], 1080-noseLoc[0:2700,1,0], s=3, c=timestamps, cmap=cmap, norm=norm)
#plt.colorbar(sc)
plt.xlim((100, 1300))
plt.ylim((-1150, 50))
plt.axis('scaled')
plt.show()



### plotting on video frames ###

video = sleap.load_video('/home/tomhagley/Documents/SLEAPProject/octagon_solo/CameraTop_2022-11-02T14-00-00.avi')
print(video.shape)
img = video[0]
img2 = img.reshape(1080, 1440)
""" plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.scatter(600, 550)
plt.show()

img3 = img2[::-1,:]
img3.shape
plt.imshow(img3, cmap='gray', vmin=0, vmax=255)
plt.scatter(600, 550)
plt.show() """

### wall centre locations ###

# MANUALLY annotated by me to be between ports on each wall
# REMEMBER wall 3 is 0 degrees, wall 1 is 90 degrees
# clockwise from top wall (0-degrees)
# this set for img3 (y axis growing up from bottom)
wall_ycoords = [1023, 884, 547, 205, 63, 208, 549, 885]
wall_xcoords = [700, 1036, 1177, 1041, 698, 360, 220, 363]

# rotated set of coordinates to fit convention of walls starting
# from rightmost = 1, going clockwise
idxs = [2,3,4,5,6,7,0,1]
wall_ycoords_2 = [wall_ycoords[i] for i in idxs]
wall_xcoords_2 = [wall_xcoords[i] for i in idxs]

# don't need this as images are plotted from top-left as origin
wall_ycoords_3 = [1080-coord for coord in wall_ycoords]
wall_xcoords_3 = wall_xcoords



plt.figure()
#plt.axes('scaled')
plt.imshow(img3, cmap='gray', vmin=0, vmax=255, origin='lower')
plt.scatter(wall_xcoords_2[0], wall_ycoords_2[0], s=3, color='y')
plt.show()

### scatter walls plus trajectory ###
plt.figure(figsize=(7,7))
plt.scatter(wall_xcoords_3, wall_ycoords_3, s=4, color='b')
plt.scatter(noseLoc[0:2700,0,:], noseLoc[0:2700,1,:], c=timestamps, cmap=cmap, norm=norm, s=3)
plt.imshow(img2, cmap='gray')
#plt.axis('scaled')
plt.show()


### lawrence gitlab wall coords ###

# These are the wrong size!
# could use open-cv findcontours instead? 
def define_wall_coords():
    wall1 = [454, 1021]
    wall2 = [217, 927]
    wall3 = [115, 679]
    wall4 = [216, 432]
    wall5 = [460, 328]
    wall6 = [710, 429]
    wall7 = [812, 685]
    wall8 = [703, 926]
    wallCoords = {1: wall1, 2: wall2, 3: wall3, 4: wall4, 5:wall5, 6:wall6, 7:wall7, 8:wall8}
    return wallCoords

def oct_corners():
    return [(545, 950), (825, 950), (1020, 755), (1020, 480), (825, 275), (545, 275), (344, 480), (344, 755), (545, 950)]

corner_coords = oct_corners()
wallCoords = define_wall_coords()
wall_xcoords3 = [value[0] for value in wallCoords.values()]
wall_ycoords3 = [val[1] for val in wallCoords.values()]
corner_xcoords = [coords[0] for coords in corner_coords]
corner_ycoords = [coords[1] for coords in corner_coords]

plt.figure(4, figsize=(7,7))
plt.imshow(img2, cmap='gray')
plt.scatter(corner_xcoords, corner_ycoords, s=4, color='b')
plt.show()

### mask and normalisation ###

# These coords are accurate and from Grayson's octpy
# can use to apply normalisation
center = (702, 536)
top_left = (227, 66)
width, height = 945, 939

octaMask = cv.imread('/home/tomhagley/Documents/SLEAPProject/docs/TrackingMask2022-10-05T14_05_40.png', cv.IMREAD_GRAYSCALE)
plt.figure(5)
plt.subplot(121)
plt.imshow(octaMask, cmap='gray')
plt.subplot(122)
plt.imshow(img2)
plt.show()

# create combined mask and image
combined = np.vstack((octaMask[:540,:], img2[540:1081,:]))
plt.figure(6, figsize=(7,7))
plt.scatter(center[0], center[1], s=10, color='b')
plt.scatter(top_left[0], top_left[1], s=10, color='r')
plt.imshow(combined, cmap='gray')
plt.show()

### bonsai data ###

# paths
directory_bonsai = '/home/tomhagley/Documents/SLEAPProject/data/bonsai_metadata'
fileName_bonsai = '2022-11-02_A006.csv'
filePath_bonsai = directory_bonsai + os.sep + fileName_bonsai

bonsaiData = pd.read_csv(filePath_bonsai)
rawX_bonsai = bonsaiData['raw_trajectory_x_1'][10]
rawY_bonsai = bonsaiData['raw_trajectory_y_1'][10]

rawX_bonsai = rawX_bonsai.split(', ')
rawX_bonsai[0] = rawX_bonsai[0][1:]
rawX_bonsai[-1] = rawX_bonsai[-1][:-1]

rawY_bonsai = rawY_bonsai.split(', ')
rawY_bonsai[0] = rawY_bonsai[0][1:]
rawY_bonsai[-1] = rawY_bonsai[-1][:-1]

xCoords_bonsai = [int(float(coord)) for coord in rawX_bonsai]
yCoords_bonsai = [int(float(coord)) for coord in rawY_bonsai]

### plot bonsai tracking data over image ###

plt.figure(7, figsize=(7,7))
plt.imshow(img2, cmap='gray')
plt.scatter(xCoords_bonsai, yCoords_bonsai, s=3, color='b')
plt.show()




### useful tricks ###
""" .values on pandas series will return a numpy array
    instead of a series.

    dividing a datetime object by np.timedelta64(1, 's')
    will return something in the form of seconds only (and as float)
    
    Remember can normalise image data using only the coordinates
    of the centre point and e.g. top-left point

    If two series have shared timestamp indices you can 
    pd.concat them and them .sort_index()

    Calculate a series as 1s and 0s based on a condition
    and them use .astype(bool) to change to boolean

    Reshape can broadcast: can reshape an array into shape
    [1,4] and it will produce a 2D array of 1x4 rows 

    To round normally (instead of flooring), just add 0.5
    to every value before casting as int 
    
    Look back at octagon probability plot code for useful 
    pandas tips s.a. using apply and indexing properly
    
    Can use pandas df.at to select specific scalar indices 
    instead of whole columns
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html
    """
