import logging
import multiprocessing
import threading

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from classification.classifier import ClassifierByFlatCoordinates, ClassifierByAngles, \
    ClassifierByAnglesAndCoordinates, ExperimentalClassifier
from feature_extraction.pre_processor import run_pre_process_steps, pre_process_single_frame, un_flatten_points
from feature_extraction.renderer import render, render_static, render_static_2_hands, render_static_and_dynamic
from pose_estimation.interfacer import mp_estimate_pose, mp_estimate_pose_static
from utils.constants import ClassificationMethods, LABEL_VS_INDEX
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
    means_file = './data/training/reference-signs-aw-01-left.csv'
    means: pd.DataFrame = pd.read_csv(means_file)

    means_file_2 = './data/training/reference-signs-aw-01-right.csv'
    means2 = pd.read_csv(means_file_2)

    means = means.append(means2)
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
    return out_put


def validate():
    all_results = pd.DataFrame()
    means = _get_training_data()
    classifier = ExperimentalClassifier(means, None)
    for file_id in range(1, 2):
        file_path = './data/labels/{}.csv'.format(file_id)
        labels: pd.DataFrame = pd.read_csv(file_path)
        labels = labels[labels['label']!=22]
        results_for_file = pd.DataFrame()
        for index, row in labels.iterrows():
            video_m = video_meta.get(file_id)
            video = video_m.get('location')
            fps = video_m.get('fps')

            start_time = row['start']
            end_time = row['end']

            start_frame = start_time * fps
            end_frame = end_time * fps

            total_frames = end_frame - start_frame

            for frame in [start_frame + total_frames * .1 * i for i in range(1, 10)]:
                if row['label'] == 50 or row['label'] == 51: continue
                logging.info('Processing a single frame...')
                image = get_static_frame(video, frame / fps, fps=fps)
                land_marks = mp_estimate_pose_static(image)
                land_marks = pre_process_single_frame(land_marks)

                # prediction = classify_static(land_marks, means, method=ClassificationMethods.ANGLES)
                prediction = classifier.classify(land_marks)
                if prediction[0].get('class') == 'NA': continue
                prediction[0].update({'truth_sign': LABEL_VS_INDEX.get(row['label'])})
                result_row = pd.DataFrame(prediction[0], index=[0])
                results_for_file = results_for_file.append(result_row)

        all_results = all_results.append(results_for_file)

        acc = accuracy_score(results_for_file['truth_sign'], results_for_file['class'])
        pre = precision_score(results_for_file['truth_sign'], results_for_file['class'], average='macro')
        logging.info('Results for the file {}\nacc {} \npre {}'.format(file_id, acc, pre))

    acc = accuracy_score(all_results['truth_sign'], all_results['class'])
    pre = precision_score(all_results['truth_sign'], all_results['class'], average='macro')
    logging.info('Results for all \nacc {} \npre {}'.format(acc, pre))
    _plot_cnf_matrix(all_results)


def _plot_cnf_matrix(all_results):
    cf_matrix = confusion_matrix(all_results['truth_sign'], all_results['class'],
                                 labels=all_results['truth_sign'].unique())

    ## Display the visualization of the Confusion Matrix.
    df_cm = pd.DataFrame(cf_matrix, index=all_results['truth_sign'].unique(),
                         columns=all_results['truth_sign'].unique())
    ax = sns.heatmap(df_cm, annot=True, cmap='Blues')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Category')
    ax.set_ylabel('Actual Category ')

    plt.show()


if __name__ == '__main__':
    logging.info('Initiating classification...')
    video_m = video_meta.get(7)
    video = video_m.get('location')
    fps = video_m.get('fps')

    time = 31.2

    # process_single_frame(video, time, fps, classify=True, method=ClassificationMethods.ANGLES)

    # process_video(video=None, classify=True, method=ClassificationMethods.ANGLES)
    # process_video()
    # means = _get_training_data()
    # classifier = DecisionTreeClassifier(means, 2)
    #
    # image = get_static_frame(video, time, fps=fps)
    # land_marks = mp_estimate_pose_static(image)
    # land_marks = pre_process_single_frame(land_marks)
    #
    # out_put = classifier.classify(land_marks)
    # logging.info(out_put)

    validate()
