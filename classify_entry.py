import logging
import multiprocessing
import threading

import pandas as pd

from classification.classifier import ClassifierByFlatCoordinates, ClassifierByAngles, \
    ClassifierByAnglesAndCoordinates
from feature_extraction.pre_processor import run_pre_process_steps, pre_process_single_frame, un_flatten_points
from feature_extraction.renderer import render, render_static, render_static_2_hands, render_static_and_dynamic
from pose_estimation.interfacer import mp_estimate_pose, mp_estimate_pose_static
from utils.constants import ClassificationMethods
from utils.video_utils import get_static_frame, show_frame, video_meta

logging.basicConfig(level=logging.INFO)

def process_video(video=None, classify=False, method=ClassificationMethods.FLAT_COORDINATES):
    no_of_processes = 3 if classify else 2
    pool = multiprocessing.Pool(processes=no_of_processes)
    m = multiprocessing.Manager()
    hand_pose_q = m.Queue()

    pool.apply_async(mp_estimate_pose, (hand_pose_q, video))
    pre_processed_q_1 = m.Queue()

    duplicate_queues = []
    if classify:
        pre_processed_q_2 = m.Queue()
        duplicate_queues = [pre_processed_q_2]
        training_data = _get_training_data()
        pool.apply_async(classifier_worker, (pre_processed_q_2, training_data, method))

    pool.apply_async(run_pre_process_steps, (hand_pose_q, pre_processed_q_1, duplicate_queues))
    # logging.error(un_flatten_points(list(training_data.drop('sign', axis=1, errors='ignore').iloc[0])))
    # render_static_and_dynamic(pre_processed_q_1,un_flatten_points(list(training_data.drop('sign', axis=1, errors='ignore').iloc[0])) )
    render(pre_processed_q_1)


def process_single_frame(video_file, seconds, fps, classify=False, method=ClassificationMethods.ANGLES_AND_FLAT_CO):
    logging.info('Processing a single frame...')
    image = get_static_frame(video_file, seconds, fps=fps)
    land_marks = mp_estimate_pose_static(image)
    land_marks = pre_process_single_frame(land_marks)

    if classify:
        means = _get_training_data()
        classify_static(land_marks, means, method=method)

    threading.Thread(target=show_frame, args=(image,), daemon=True).start()

    render_static(land_marks)
    # render_static_2_hands(land_marks, land_marks_2)


def _get_training_data():
    # TODO: Rename
    means_file = './data/training/means_cham_vertices_28_10_21_2_i-replaced.csv'
    means = pd.read_csv(means_file)
    return means


def classifier_worker(processed_q, means, method):
    logging.info('Classifier worker running. Method: {}'.format(method))
    if method == ClassificationMethods.FLAT_COORDINATES:
        classifier = ClassifierByFlatCoordinates(means, 2, [])
    if method == ClassificationMethods.ANGLES:
        classifier = ClassifierByAngles(means)
    if method == ClassificationMethods.ANGLES_AND_FLAT_CO:
        classifier = ClassifierByAnglesAndCoordinates(means, 2)
    previous_sign = None
    while True:
        try:
            if not processed_q.empty():
                vertices = processed_q.get()
                candidate_signs = classifier.classify(vertices)
                if candidate_signs and candidate_signs[0]['class'] != previous_sign:
                    logging.info('Classification result {}'.format(candidate_signs))
                    previous_sign = candidate_signs[0]['class']
            else:
                pass
        except Exception as e:
            logging.error(e)
            break


def classify_static(land_marks, means, method=ClassificationMethods.FLAT_COORDINATES):
    logging.info('Classifying an static image. Method: {}'.format(method))
    if method == ClassificationMethods.FLAT_COORDINATES:
        # classifier = ClassifierByFlatCoordinates(means, vertices_to_ignore=[0, 5, 9, 13, 17])
        classifier = ClassifierByFlatCoordinates(means, 2)
    elif method == ClassificationMethods.ANGLES:
        classifier = ClassifierByAngles(means)
    elif method == ClassificationMethods.ANGLES_AND_FLAT_CO:
        classifier = ClassifierByAnglesAndCoordinates(means, 2)

    out_put = classifier.classify(land_marks)
    logging.info(out_put)


if __name__ == '__main__':
    logging.info('Initiating classification...')
    video_m = video_meta.get(4)
    video = video_m.get('location')
    fps = video_m.get('fps')

    time = 4*60 + 16

    process_single_frame(video, time, fps, classify=True, method=ClassificationMethods.ANGLES_AND_FLAT_CO)

    # process_video(video, classify=True, method=ClassificationMethods.ANGLES_AND_FLAT_CO)
