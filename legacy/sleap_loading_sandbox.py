import sleap
from sleap.io.video import Video, MediaVideo
from sleap.io.visuals import img_to_cv, save_labeled_video
from matplotlib import pyplot as plt
import cv2 as cv


### Example dataset usage ###

# paths
video_filename = 'CameraTop_2022-11-02T14-00-00.avi_model5_predictions_230206_CLI.slp'
labels_filename = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02T14-00-00.avi_model5_predictions_230206_CLI.slp' 

# high level interface class for all data making up an annotated project
labels = sleap.load_file(labels_filename)
video = labels.video
                                          
# identify a specific list of labeledFrames like this:
labels_subset = labels.find(video, range(5, 10))

# list of all instance.labeledFrame objects in the labels dataset
labeledFrames = labels.labeled_frames

# use .frames() method to return an iterator/generator instead of an iterable
frames2 = labels.frames(video)

# list of identified tracks in labeled labeledFrames
tracks = labels.tracks

# single instance.LabeledFrame object
labeledFrame = labeledFrames[0]

# try to find given track in this LabeledFrame
found_track = labeledFrame.find(tracks[0])

# image associated with this LabeledFrame
img = labeledFrame.image
# instances associated with this LabeledFrame
instances = labeledFrame.instances

# misc useful LabeledFrame methods
print(labeledFrame.has_predicted_instances)
print(labeledFrame.has_tracked_instances)
print(labeledFrame.has_user_instances)
print(labeledFrame.frame_idx)

# display the image for this labeledFrame
# in mpl
plt.figure(1)
plt.imshow(img, cmap='gray')
plt.show()
# in open-cv
img_cv = sleap.io.visuals.img_to_cv(img)
cv.imshow('window1', img_cv)
cv.waitKey()

# points_array associated with the first instance
points_array = instances[0].points_array
# alternatively access points directly
points = instances[0].points

# example: extract nose coordinates from each point
# using points directly lets you access the visibility
noseCoords_points_array = (points_array[0,0], points_array[0,1])
noseCoords_points = (points[0].x, points[0].y, points[0].visible)


# save video of trial with annotations, then plot first and last frame
sleap.io.visuals.save_labeled_video('/home/tomhagley/Documents/SLEAPProject/octagon_solo/test.avi', labels, video, \
        frames=list(range(0,501)), fps=13, marker_size=2, show_edges=True)
labeledFrame_first = labeledFrames[0]
labeledFrame_last = labeledFrames[50]

# plot individual annotated frames
labeledFrame_first.plot()
plt.show()
labeledFrame_last.plot()
plt.show()


# play trial video with opencv
cap = cv.VideoCapture('/home/tomhagley/Documents/SLEAPProject/octagon_solo/test.avi')

# Read until video is completed
while(cap.isOpened()):
      
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        cv.imshow('Frame', frame)
          
    # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
  
# Break the loop
    else:
        break
  
# When everything done, release
# the video capture object
cap.release()
  
# Closes all the frames
cv.destroyAllWindows()







### Attempted path change ###

# sleap.io.pathutils.filenames_prefix_change([video_filename], '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions', \
#                                     '/home/tomhagley/Documents/sleap_project/octagon_solo/predictions')
# path = video.fixup_path('/home/tomhagley/Documents/sleap_project/octagon_solo' + os.sep + video_filename)
