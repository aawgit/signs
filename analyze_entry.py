import pandas as pd

from feature_extraction.pre_processor import scale_vertices2
from feature_extraction.renderer import render_static, render_static_2_hands
from feature_extraction.pre_processor import pre_process_single_frame
from pose_estimation.interfacer import mp_estimate_pose_static
from pose_estimation.pose_estimator_by_frame import get_static_frame
from utils.video_utils import video_meta


def get_saved_land_mark(sign, sign_file_df):
    coordinates = list(sign_file_df[sign_file_df['sign'] == sign].iloc[0])[1:]
    land_mark = [tuple(coordinates[i:i + 3]) for i in range(0, len(coordinates), 3)]
    return land_mark

if __name__ == '__main__':
    means_file = './data/training/reference_signs_1-21-11-21.csv'
    means = pd.read_csv(means_file)

    means_file_2 = './data/training/reference_signs_21-11-21.csv'
    means2 = pd.read_csv(means_file_2)

    video_m = video_meta.get(4)
    video = video_m.get('location')
    fps = video_m.get('fps')

    time = 4*60 + 16

    saved_lm = get_saved_land_mark(16, means)
    saved_lm = pre_process_single_frame(saved_lm)
    # render_static(saved_lm)

    saved_lm2 = get_saved_land_mark(16, means2)
    # saved_lm = pre_process_single_frame(saved_lm)

    image = get_static_frame(video, time, fps=fps)
    land_marks = mp_estimate_pose_static(image)
    land_marks = pre_process_single_frame(land_marks)

    # render_static(land_marks)

    render_static_2_hands(land_marks, saved_lm2)