import csv
import logging
import multiprocessing
import threading
import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance

from src.pose_estimation.vertices_mapper import EDGES_MEDIA_PIPE
from src.utils.video_utils import get_static_frame2

from src.feature_extraction.pre_processor import run_pre_process_steps, pre_process_single_frame, flatten_points, \
    un_flatten_points
from src.feature_extraction.renderer import render
from src.pose_estimation.interfacer import mp_estimate_pose_static, mp_estimate_pose_from_image

from src.utils.constants import ClassificationMethods
from src.utils.video_utils import get_static_frame
from src.classification.classify_entry import get_training_data, classifier_worker, is_wrong_estimate

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

    training_data = get_training_data()
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
    if not target_file_name: target_file_name = data_set_id

    for index, row in labels.iterrows():
        label = row['label']
        file_name = row['file_name']

        if label == 51:
            continue
        logging.info('Processing a single frame...')

        land_marks = mp_estimate_pose_from_image(image_file_path+file_name)
        if not land_marks:  continue
        land_marks, angles = pre_process_single_frame(land_marks)
        if not land_marks: continue
        print('Saving current landmark {}'.format(land_marks))

        path = "./data/training/{}.csv".format(str(target_file_name).split(".")[0])
        lm_row = []
        for landmark_point in land_marks:
            lm_row.extend(np.round(landmark_point, 4))
        with open(path, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([label, *lm_row, data_set_id])

def save_landmark_plot(label: str, landmark: list, plot_folder: str, file_name: str):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(projection='3d'))
    views = [(-90, 90), (0, 90), (0, 0), (45, 90)]
    for i, ax in enumerate(axs.flat):
        # ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(0, 2)
        ax.set_zlim3d(-1, 1)
        ax.view_init(*views[i])
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        un_flattened = un_flatten_points(landmark)
        zdata = [point[2] for point in un_flattened]
        xdata = [point[0] for point in un_flattened]
        ydata = [point[1] for point in un_flattened]
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

        for point_pair in EDGES_MEDIA_PIPE:
            first = point_pair[0]
            second = point_pair[1]
            ax.plot3D([xdata[first], xdata[second]], [ydata[first], ydata[second]], [zdata[first], zdata[second]], 'b')
    plt.savefig("{}/{}-{}.png".format(plot_folder, label, file_name))

def plot_images(landmarks: pd.DataFrame):
    i = 0
    for row in landmarks.values.tolist():
        save_landmark_plot(row[0], row[1: 64], '../_TMP', i)
        i = 1 + i

def save_landmark_plot_tmp(label: str, landmark: list, plot_folder: str, file_name: str):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(0, 1.5)
    ax.set_zlim3d(-1, 1)
    ax.view_init(-60, 90)
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    un_flattened = un_flatten_points(landmark)
    zdata = [point[2] for point in un_flattened]
    xdata = [point[0] for point in un_flattened]
    ydata = [point[1] for point in un_flattened]
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')

    for point_pair in EDGES_MEDIA_PIPE:
        first = point_pair[0]
        second = point_pair[1]
        ax.plot3D([xdata[first], xdata[second]], [ydata[first], ydata[second]], [zdata[first], zdata[second]], 'b')
    plt.savefig("{}/{}-{}.png".format(plot_folder, label, file_name))

def save_lm_polot_temp(file):
    try:
        landmarks = mp_estimate_pose_from_image(file)
    except Exception as e:
        logging.error("Error in getting estimation for image {}".format(file))
        return
    if not landmarks:
        logging.warning("A landmark was not found for {}".format(file))
        return
    landmarks, original_angles = pre_process_single_frame(landmarks)
    lm_row = []
    for landmark in landmarks:
        lm_row.extend(np.round(landmark, 4))

    save_landmark_plot_tmp('5', lm_row, '/home/aka/Downloads/', 'image')


def save_landmark_and_plot_images(data_set_id) -> None:
    base_directory = '../data'
    labels_file = '{}/{}/labels.csv'.format(base_directory, data_set_id)
    images_folder = '{}/{}/images'.format(base_directory, data_set_id)
    output_path = "{}/{}/{}.csv".format(base_directory, data_set_id, 'landmarks')
    plot_folder = "{}/{}/plots".format(base_directory, data_set_id)

    labels = pd.read_csv(labels_file)

    for index, row in labels.iterrows():
        file_name = row["file_name"]
        label = row["label"]
        try:
            landmarks = mp_estimate_pose_from_image("{}/{}".format(images_folder, file_name))
        except Exception as e:
            logging.error("Error in getting estimation for image {}".format(file_name))
            continue
        if not landmarks:
            logging.warning("A landmark was not found for {}".format(file_name))
            continue
        landmarks, original_angles = pre_process_single_frame(landmarks)

        lm_row = []
        for landmark in landmarks:
            lm_row.extend(np.round(landmark, 4))

        save_landmark_plot(label, lm_row, plot_folder, file_name.split('.')[0])

        with open(output_path, 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow([label, *lm_row, data_set_id, file_name])

        # plt.savefig("{}/{}-{}.png".format(plot_folder, label, file_name.split('.')[0]))

def save_landmark_and_plot_video(subject_id, n_samples):
    base_directory = '../data'
    labels_file = '{}/{}/labels.csv'.format(base_directory, subject_id)
    video_file = '{}/{}/video.mp4'.format(base_directory, subject_id)
    output_path = "{}/{}/{}.csv".format(base_directory, subject_id, 'landmarks')
    plot_folder = "{}/{}/plots".format(base_directory, subject_id)

    we_file_path = '../../data/incorrect_estimates.csv'
    wrong_estimates_df = pd.read_csv(we_file_path)

    labels: pd.DataFrame = pd.read_csv(labels_file)

    for index, row in labels.iterrows():
        label = row['label']
        if is_wrong_estimate(subject_id, label, wrong_estimates_df): continue
        if label == 50: continue
        start_frame = row['start']
        end_frame = row['end']

        total_frames = end_frame - start_frame
        for frame in [start_frame + total_frames * .25 * i for i in [1, 3]]:
            image = get_static_frame2(video_file, frame)
            land_marks = mp_estimate_pose_static(image)
            if not land_marks: continue
            landmarks, angles = pre_process_single_frame(land_marks)
            lm_row = []
            for landmark in landmarks:
                lm_row.extend(np.round(landmark, 4))

            save_landmark_plot(label, lm_row, plot_folder, frame)

            with open(output_path, 'a') as fd:
                writer = csv.writer(fd)
                writer.writerow([label, *lm_row, subject_id, frame])

def detect_similar_landmarks(threshold):
    training_data = get_training_data(with_origins=True)
    training_data.reset_index(inplace=True, drop=True)
    all_distances = {}
    for index, row in training_data.drop(['subject', 'image', 'correct', 'use', 'distances'], errors='ignore', axis=1).iterrows():
        sign = row['sign']
        training_data['distances'] = training_data.drop(['sign', 'subject', 'image', 'correct', 'use', 'distances'], errors='ignore', axis=1) \
            .apply(lambda x: distance.euclidean(x, row[1:]), axis=1)
        small_distance = training_data[training_data['distances']<threshold]
        if small_distance.shape[0]>1:
            all_distances.update({index: {'distances': small_distance[['sign', 'distances', 'subject', 'image']], 'sign': sign}})
    x = 5

if __name__ == '__main__':
    # video_m = video_meta.get(9)
    # video = video_m.get('location')
    # fps = video_m.get('fps')
    #
    # # image_and_estimation(video, callback, "new-labels-001.csv")
    # # create_training_data_from_video(video_m, 3)
    # create_training_data_from_images(11)
    # save_landmark_and_plot_images('subject09')
    # plot_images(_get_training_data_old())
    # save_landmark_and_plot_video('subject07', 3)
    # detect_similar_landmarks(0.6)
    save_lm_polot_temp('/home/aka/Downloads/Screenshot from 2022-03-29 01-02-19.png')
    # pass