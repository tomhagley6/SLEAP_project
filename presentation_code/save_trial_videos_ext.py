from utils.video_functions import save_video_trial
from session import Session

root = '/home/tomhagley/Documents/SLEAPProject'
sessions = ['2023-01-10_ADU003_ADU005']
sessions_list = [Session(root, session, social=True) for session in sessions]

# extract data for all sessions
for session in sessions_list:
    session.data_filtering('all')
    session.extract_basic_data()

trial = 35

for session in sessions_list:
    save_video_trial(session.video_path, session.grey_stims, session.grey_ends, trial, labels_file=session.labels_file,
                    video_filename= session.video_filename, colorVideo=False, fps=40, 
                    color_frame_offset=session.color_frame_offset)