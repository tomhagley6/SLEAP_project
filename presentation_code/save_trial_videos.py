from utils.video_functions import save_video_trial
from session import Session

root = '/home/tomhagley/Documents/SLEAPProject'
sessions = ['2022-11-02_A006']
sessions_list = [Session(root, session) for session in sessions]

# extract data for all sessions
for session in sessions_list:
    session.data_filtering('all')
    session.extract_basic_data()
            
trial = 20
            
save_video_trial(session.video_path, session.color_stims, session.color_ends, trial, labels_file=None,
                    video_filename= session.color_video_filename, colorVideo=True, fps=40, 
                    color_frame_offset=session.color_frame_offset)


