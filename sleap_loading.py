import sleap
from sleap.io.visuals import img_to_cv, save_labeled_video
from matplotlib import pyplot as plt
import cv2 as cv

""" Functions to extract videos and specific frames from SLEAP annotated videos """

def video_from_frame_range(labelsPath, outputPath, startFrame, stopFrame):
    """ Extract an annotated video over the given frames
        Save this video to outputPath and display """

    labels = sleap.load_file(labelsPath)
    video = labels.video

    # save video of trial with annotations, then plot first and last frame
    sleap.io.visuals.save_labeled_video(outputPath, labels, video, \
            frames=list(range(startFrame, stopFrame)), fps=50, marker_size=2, show_edges=True)

    # play trial video with opencv
    cap = cv.VideoCapture(outputPath)

    # Read until video is completed
    while(cap.isOpened()):
        
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
        # Display the resulting frame
            cv.imshow('Frame', frame)
            
            key = cv.waitKey(1)
        # Press Q on keyboard to exit
            # if cv.waitKey(25) & 0xFF == ord('q'):
            if key == ord('q'):
                break
        # Press P on keyboard to pause
            if key == ord('p'):
                cv.waitKey(-1) # wait for any key press
    
        else:
            break

        cv.waitKey(20)
    
    # When done, release video capture obj
    cap.release()
    
    # Closes all the frames
    cv.destroyAllWindows()

    return None


def images_from_frame_range(labelsPath, startFrame, stopFrame):
    """ Extract annotated frames from the beginning and end
        of a video chunk and display them """
    
    # load the labels file and find subset of LabeledFrames
    labels = sleap.load_file(labelsPath)
    video = labels.video
    labels_subset = labels.find(video, range(startFrame, stopFrame+1))

    labeledFrame_first = labels_subset[0]
    labeledFrame_last = labels_subset[-1]
    
    # plot
    labeledFrame_first.plot()
    plt.show()
    labeledFrame_last.plot()
    plt.show()

    return None


if __name__ == '__main__':
    """ Show video within frame range and images from start and end of frame range """
    outputPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/test.avi'
    labelsPath = '/home/tomhagley/Documents/SLEAPProject/octagon_solo/predictions/CameraTop_2022-11-02T14-00-00.avi_model5_predictions_230206_CLI.slp' 
    startFrame = 7383
    stopFrame = 7441
    video_from_frame_range(labelsPath=labelsPath, outputPath=outputPath, startFrame=startFrame, stopFrame=stopFrame)
    images_from_frame_range(labelsPath=labelsPath, startFrame=startFrame, stopFrame=stopFrame)
