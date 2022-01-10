import numpy as np
import pandas as pd

import cv2
from feature_extraction.pre_processor import scale_vertices2, flatten_points
from feature_extraction.renderer import render_static, render_static_2_hands
from feature_extraction.pre_processor import pre_process_single_frame
from pose_estimation.interfacer import mp_estimate_pose_static
from pose_estimation.pose_estimator_by_frame import get_static_frame
from classify_entry import _get_training_data
from utils.video_utils import video_meta


def get_saved_land_mark(sign, sign_file_df, source=None):
    if not source:
        filtered_df = sign_file_df[sign_file_df['sign'] == sign].drop('source', axis=1, errors='ignore')
    else:
        filtered_df = sign_file_df[(sign_file_df['sign'] == sign) & (sign_file_df['source'] == source)]. \
            drop('source',
                 axis=1,
                 errors='ignore')
    coordinates = list(filtered_df.iloc[6])[1:]
    land_mark = [tuple(coordinates[i:i + 3]) for i in range(0, len(coordinates), 3)]
    return land_mark


if __name__ == '__main__':
    means = _get_training_data()

    video_m = video_meta.get(1)
    video = video_m.get('location')
    fps = video_m.get('fps')

    time = 86  # testing th, estimations seem flat

    saved_lm = get_saved_land_mark(27, means)
    saved_lm = pre_process_single_frame(saved_lm)
    render_static(saved_lm)
    #
    # saved_lm2 = get_saved_land_mark(16, means2)

    image = get_static_frame(video, time, fps=fps)
    # image = cv2.imread('./data/video/ref/left/51_3.jpg')
    land_marks = mp_estimate_pose_static(image)
    land_marks = pre_process_single_frame(land_marks)
    # #
    # flatted = flatten_points(land_marks)
    # rounded = [np.round(p, 4) for p in flatted]
    # print(rounded)
    #
    # video_m = video_meta.get(2)
    # video = video_m.get('location')
    # fps = video_m.get('fps')

    # time = 1 + 48
    # image2 = get_static_frame(video, time, fps=fps)
    # land_marks2 = mp_estimate_pose_static(image2)
    # land_marks2 = pre_process_single_frame(land_marks2)

    # render_static(land_marks)

    # render_static_2_hands(land_marks, saved_lm)
