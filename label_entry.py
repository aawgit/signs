import csv
import logging
import multiprocessing
import threading
import os

import numpy as np
import pandas as pd
import cv2

from utils.video_utils import video_meta
from pose_estimation.interfacer import mp_estimate_pose_static

from feature_extraction.pre_processor import run_pre_process_steps, pre_process_single_frame, flatten_points
from feature_extraction.renderer import render, render_static, render_static_2_hands, render_static_and_dynamic
from pose_estimation.interfacer import mp_estimate_pose, mp_estimate_pose_static, mp_estimate_pose_from_image
from pose_estimation.media_pipe_static_estimator import static_images

from utils.constants import ClassificationMethods
from utils.video_utils import get_static_frame, show_frame, video_meta
from classify_entry import _get_training_data, classifier_worker

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


class LabellerWindow:
    def __init__(self, video, callback, hand_pose_q, file_name=None):
        self.video = video
        self.callback = callback
        self.queue = hand_pose_q
        self.current_landmark = []
        self.file_name = file_name

    def update_current_landmark(self, land_marks):
        self.current_landmark = land_marks

    def save_current_landmark(self, *args):
        print('Saving current landmark {}'.format(self.current_landmark))
        if not self.file_name: self.file_name = os.path.basename(self.video)
        path = "./data/training/{}.csv".format(self.file_name.split(".")[0])
        row = []
        lm = pre_process_single_frame(self.current_landmark)
        for landmark_point in lm:
            row.extend(np.round(landmark_point, 4))
        with open(path, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(['<SET>', *row])

    def process_callback_wrapper(self):
        pass

    def on_change_wrapper(self, trackbarValue):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
        err, img = self.cap.read()
        cv2.imshow("Video feed", img)
        land_marks = self.callback(img, trackbarValue, self.queue)
        self.update_current_landmark(land_marks)

    def video_position_selector(self):
        logging.info('Starting video position selector...')
        cap = cv2.VideoCapture(self.video)
        self.cap = cap
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow("Video feed", cv2.WINDOW_NORMAL)
        cv2.createButton("Save", self.save_current_landmark, None, cv2.QT_PUSH_BUTTON)

        cv2.createTrackbar('start', 'Video feed', 0, length, self.on_change_wrapper)
        cv2.createTrackbar('end', 'Video feed', 100, length, self.on_change_wrapper)

        self.on_change_wrapper(0)
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


def image_and_estimation(video, callback, fname=None):
    pool = multiprocessing.Pool(processes=2)
    m = multiprocessing.Manager()

    hand_pose_q = m.Queue()
    pre_processed_q_1 = m.Queue()
    pre_processed_q_2 = m.Queue()
    pre_processed_q_3 = m.Queue()

    pool.apply_async(run_pre_process_steps, (hand_pose_q, pre_processed_q_1, [pre_processed_q_2, pre_processed_q_3]))

    pool.apply_async(render, (pre_processed_q_1,))

    threading.Thread(target=display_coordinates, args=(pre_processed_q_2,), daemon=True).start()

    training_data = _get_training_data()
    pool.apply_async(classifier_worker, (pre_processed_q_3, training_data, ClassificationMethods.ANGLES))

    lw = LabellerWindow(video, callback, hand_pose_q, fname)
    lw.video_position_selector()


def display_coordinates(queue):
    while True:
        try:
            if not queue.empty():
                current_vertices = queue.get()
                logging.info(flatten_points(current_vertices))
        except Exception as e:
            break


def callback(img, frame_no, queue):
    logging.info(frame_no)
    if img is None: return
    land_marks = mp_estimate_pose_static(img)
    if land_marks and len(land_marks) > 0:
        queue.put((land_marks, frame_no))
        return land_marks


def create_training_data_from_video(video_m, file_id, file_name=None):
    file_path = './data/labels/{}.csv'.format(file_id)
    labels: pd.DataFrame = pd.read_csv(file_path)

    for index, row in labels.iterrows():
        video = video_m.get('location')
        fps = video_m.get('fps')

        start_time = row['start']
        end_time = row['end']

        start_frame = start_time * fps
        end_frame = end_time * fps

        total_frames = end_frame - start_frame

        for frame in [start_frame + total_frames * .25 * i for i in [1, 3]]:
            if row['label'] == 50 or row['label'] == 51: continue
            logging.info('Processing a single frame...')
            image = get_static_frame(video, frame / fps, fps=fps)
            land_marks = mp_estimate_pose_static(image)
            land_marks = pre_process_single_frame(land_marks)
            print('Saving current landmark {}'.format(land_marks))
            if not file_name: file_name = os.path.basename(video)
            path = "./data/training/{}.csv".format(file_name.split(".")[0])
            sign = row['label']
            lm_row = []
            for landmark_point in land_marks:
                lm_row.extend(np.round(landmark_point, 4))
            with open(path, 'a') as fd:
                writer = csv.writer(fd)
                writer.writerow([sign, *lm_row])


def create_training_data_from_images(data_set_id, target_file_name=None):
    file_path = './data/labels/{}.csv'.format(data_set_id)
    labels: pd.DataFrame = pd.read_csv(file_path)

    image_file_path = './data/images/{}/'.format(data_set_id)

    for index, row in labels.iterrows():
        label = row['label']
        file_name = row['file_name']

        if label == 50 or label == 51:
            continue
        logging.info('Processing a single frame...')

        land_marks = mp_estimate_pose_from_image(image_file_path+file_name)
        land_marks = pre_process_single_frame(land_marks)
        if not land_marks: continue
        print('Saving current landmark {}'.format(land_marks))
        if not target_file_name: target_file_name = data_set_id
        path = "./data/training/{}.csv".format(str(target_file_name).split(".")[0])
        lm_row = []
        for landmark_point in land_marks:
            lm_row.extend(np.round(landmark_point, 4))
        with open(path, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([label, *lm_row, data_set_id])


if __name__ == '__main__':
    video_m = video_meta.get(5)
    video = video_m.get('location')
    fps = video_m.get('fps')

    # image_and_estimation(video, callback, "reference-signs-aw-01-right-.csv")
    # create_training_data(video_m, 3)
    create_training_data_from_images(9)