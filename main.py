import logging
import multiprocessing
import pandas as pd

from feature_extraction.pre_processor import run_pre_process_steps, pre_process_single_frame
from feature_extraction.renderer import render, render_static, render_static_2_hands
from pose_estimation.interfacer import mp_estimate_pose
from pose_estimation.pose_estimator_by_frame import get_estimation_for_frame
from feature_extraction.data_builder import write_labeled_landmarks_to_csv
from classification.classifier import classifier_by_vertices_worker, classify_land_mark

logging.getLogger().setLevel(logging.INFO)


def run_landmark_viewer(video=None):
    pool = multiprocessing.Pool(processes=2)
    m = multiprocessing.Manager()
    hand_pose_q = m.Queue()

    pose_estimation_worker = pool.apply_async(mp_estimate_pose, (hand_pose_q, video))
    pre_processed_q_1 = m.Queue()
    pre_processed_q_2 = m.Queue()

    pre_processing_worker = pool.apply_async(run_pre_process_steps, (hand_pose_q, pre_processed_q_1, pre_processed_q_2))
    render(pre_processed_q_1)


def run_classification(means, video=None):
    # video = "./data/video/WhatsApp Video 2021-10-23 at 12.51.45.mp4"
    # file = "/home/aka/Downloads/ego hands dataset/videos_1/Subject04/Scene2/Color/rgb2.avi"

    pool = multiprocessing.Pool(processes=3)
    m = multiprocessing.Manager()
    hand_pose_q = m.Queue()

    pose_estimation_worker = pool.apply_async(mp_estimate_pose, (hand_pose_q, video))

    pre_processed_q_1 = m.Queue()
    pre_processed_q_2 = m.Queue()

    output_file = "./data/training/flatten_vertices_with_labels.csv"
    pre_processing_worker = pool.apply_async(run_pre_process_steps, (hand_pose_q, pre_processed_q_1, pre_processed_q_2))
    # data_writer_worker = pool.apply_async(write_labeled_landmarks_to_csv, (pre_processed_q_2, output_file))


    classifier_by_vertices_worker(pre_processed_q_1, means)
    # workers_3 = pool.apply_async(classifier_by_vertices_worker, (pre_processed_q_1, means))
    #
    # render(pre_processed_q_1)


def get_static_land_mark(sign, means):
    coordinates = list(means[means['sign'] == sign].iloc[0])[1:]
    land_mark = [tuple(coordinates[i:i + 3]) for i in range(0, len(coordinates), 3)]
    return land_mark


def run_static_viewer(sign, means):
    # TO render a single sign
    # means_file = './data/training/means_cham_vertices_19_10_21_1.csv'
    coordinates = list(means[means['sign'] == sign].iloc[0])[1:]
    land_mark = [tuple(coordinates[i:i + 3]) for i in range(0, len(coordinates), 3)]
    render_static(land_mark)


def get_land_marks_for_frame(video, video_play_sec):
    land_mark = get_estimation_for_frame(video, video_play_sec)
    land_mark = pre_process_single_frame(land_mark)
    return land_mark


def classify_static(land_marks, means):
    out_put = classify_land_mark(land_marks, means)
    logging.info(out_put)


if __name__ == '__main__':
    # video = "./data/video/SLSL - Sinhala Sign Alphabet - Sri Lankan Sign Language - Chaminda Hewapathirana.mp4"
    # file = None
    video = "./data/video/yt1s.com - SLSL1Sinhala Manual Alphabet_360p.mp4"
    # video = "./data/video/WhatsApp Video 2021-10-23 at 12.51.45.mp4"
    # file = "/home/aka/Downloads/ego hands dataset/videos_1/Subject04/Scene2/Color/rgb2.avi"
    # run_landmark_viewer()

    means_file = './data/training/means_cham_vertices_28_10_21_2_i-replaced.csv'
    means = pd.read_csv(means_file)
    # run_static_viewer(24, means)
    # run_classification(means, video)
    #
    land_marks_1 = get_land_marks_for_frame(video, (6*60 + 12))
    land_marks_2 = get_static_land_mark(24, means)
    # # run_classification(means, video)
    classify_static(land_marks_1, means)
    # out_put = classify_land_mark(land_marks_1, means)
    # logging.info(out_put)
    render_static_2_hands(land_marks_1, land_marks_2)
