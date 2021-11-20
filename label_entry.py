import csv
import logging
import multiprocessing

import numpy as np
import pandas as pd
import cv2

from utils.video_utils import video_meta
from pose_estimation.interfacer import mp_estimate_pose_static

from feature_extraction.pre_processor import run_pre_process_steps, pre_process_single_frame, un_flatten_points
from feature_extraction.renderer import render, render_static, render_static_2_hands, render_static_and_dynamic
from pose_estimation.interfacer import mp_estimate_pose, mp_estimate_pose_static
from utils.constants import ClassificationMethods
from utils.video_utils import get_static_frame, show_frame, video_meta

logging.basicConfig(level=logging.INFO)


def write_frame_vs_landmarks_to_csv(processed_q, output_file, no_of_features=21):
    logging.info('Writing to file {}'.format(output_file))
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", *[i for i in range(0, no_of_features)]])
        while True:
            try:
                if not processed_q.empty():
                    current_vertices, frame_no = processed_q.get()
                    writer.writerow([frame_no, *current_vertices])
                else:
                    pass
            except Exception as e:
                logging.error(e)
                break


def add_labels_to_landmarks(land_marks_vs_frame, label_vs_second_file, frame_rate=29.97):
    """

    :param land_marks_vs_frame_file:
    :param output_file:
    :return:
    """
    # TODO: Add dynamic or not

    label_vs_second_file.start = label_vs_second_file.apply(lambda row: int(row.start * frame_rate), axis=1)
    label_vs_second_file.end = label_vs_second_file.apply(lambda row: int(row.end * frame_rate), axis=1)

    label_vs_frame = label_vs_second_file
    # TODO: More efficient way of merging the two dfs
    sign_count = len(label_vs_frame.sign)

    def get_label(t):
        filtered_id_series = ((t >= label_vs_frame.start) & (t <= label_vs_frame.end))
        if filtered_id_series.any():
            label_idx = filtered_id_series.dot(np.arange(sign_count))
            return int(label_vs_frame.sign[label_idx])

    land_marks_vs_frame["sign"] = land_marks_vs_frame.frame.transform(get_label)
    land_marks_vs_frame = land_marks_vs_frame[land_marks_vs_frame['sign'].notna()]
    merged = pd.merge(land_marks_vs_frame, label_vs_frame, on='sign', how='left').drop(['start', 'end'], axis=1)
    return merged


def video_position_selector(video_file, process_callback, queue):
    logging.info('Starting video position selector...')
    cap = cv2.VideoCapture(video_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow("Video feed", cv2.WINDOW_NORMAL)

    def onChange(trackbarValue):
        cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
        err, img = cap.read()
        cv2.imshow("Video feed", img)
        process_callback(img, trackbarValue, queue)
        pass

    cv2.createTrackbar('start', 'Video feed', 0, length, onChange)
    cv2.createTrackbar('end', 'Video feed', 100, length, onChange)

    onChange(0)
    cv2.waitKey()

    start = cv2.getTrackbarPos('start', 'mywindow')
    end = cv2.getTrackbarPos('end', 'mywindow')
    if start >= end:
        raise Exception("start must be less than end")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    while cap.isOpened():
        err, img = cap.read()
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end:
            break
        cv2.imshow("Video feed", img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break


def image_and_estimation(video, callback):
    pool = multiprocessing.Pool(processes=2)
    m = multiprocessing.Manager()

    hand_pose_q = m.Queue()
    pre_processed_q_1 = m.Queue()
    pre_processed_q_2 = m.Queue()

    pool.apply_async(run_pre_process_steps, (hand_pose_q, pre_processed_q_1, [pre_processed_q_2]))

    pool.apply_async(render, (pre_processed_q_1,))

    video_position_selector(video, callback, hand_pose_q)


def callback(img, frame_no, queue):
    logging.info(frame_no)
    if img is None: return
    land_marks = mp_estimate_pose_static(img)
    if land_marks and len(land_marks) > 0:
        queue.put((land_marks, frame_no))


if __name__ == '__main__':
    video_m = video_meta.get(1)
    video = video_m.get('location')
    fps = video_m.get('fps')

    image_and_estimation(video, callback)
