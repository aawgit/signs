import logging
import multiprocessing
import threading
import statistics
import time as tm_mod
import json
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, classification_report

from src.classification.classifier import ClassifierByFlatCoordinates, ClassifierByAngles, \
    ClassifierByAnglesAndCoordinates, CascadedClassifier, rule_based_classify, IndividualClassifier, \
    tune_hpp_classifier_option, LogisticRegressionPoseClassifier, KNNPoseClassifier, RandomForestPoseClassifier, \
    XGBoostPoseClassifier, DesicionTreePoseClassifier, NaiveBaysPoseClassifier
from src.classification.classifier_alternative_options import CascadedClassifierAngles
from src.feature_extraction.pre_processor import run_pre_process_steps, pre_process_single_frame
from src.feature_extraction.renderer import render, render_static
from src.pose_estimation.interfacer import mp_estimate_pose, mp_estimate_pose_static, mp_estimate_pose_from_image
from src.classification.classifier import plot_cnf_matrix
from src.pose_estimation.media_pipe_dynamic_estimator import _temp_dynamic_images
from src.utils.constants import ClassificationMethods, LABEL_VS_INDEX
from src.utils.helper import OutputFilter
from src.utils.video_utils import get_static_frame, show_frame, video_meta, get_static_frame2

logging.basicConfig(level=logging.INFO)


def process_video(video=None, classify=False, is_video=False):
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
        training_data = get_training_data()
        pool.apply_async(cascaded_classifier_worker, (pre_processed_q_2, training_data, is_video))

    pool.apply_async(run_pre_process_steps, (hand_pose_q, pre_processed_q_1, duplicate_queues))
    # logging.error(un_flatten_points(list(training_data.drop('sign', axis=1, errors='ignore').iloc[0])))
    # render_static_and_dynamic(pre_processed_q_1,un_flatten_points(list(training_data.drop('sign', axis=1, errors='ignore').iloc[0])) )
    render(pre_processed_q_1)
    # sleep(600)


def process_single_frame(video_file, seconds, fps, classify=False, method=ClassificationMethods.ANGLES_AND_FLAT_CO):
    logging.info('Processing a single frame...')
    image = get_static_frame(video_file, seconds, fps=fps)
    land_marks = mp_estimate_pose_static(image)
    land_marks = pre_process_single_frame(land_marks)

    if classify:
        means = get_training_data()
        classify_static(land_marks, means, method=method)

    threading.Thread(target=show_frame, args=(image,), daemon=True).start()

    render_static(land_marks)
    # render_static_2_hands(land_marks, land_marks_2)


def _temp_find_frame():
    means = _get_training_data_old()

    video = '../data/subject01/videos/right/th-w.mp4'

    _temp_dynamic_images(means, video)


def _get_training_data_old():
    # TODO: Rename
    means_file = '../../data/training/reference-signs-aw-01-right.csv'
    means: pd.DataFrame = pd.read_csv(means_file)

    means_file_2 = '../../data/training/reference-signs-aw-01-left.csv'
    means2 = pd.read_csv(means_file_2)
    means = means.append(means2)

    # means_file_3 = './data/training/reference-signs-geshani.csv'
    # means3 = pd.read_csv(means_file_3)
    # means = means.append(means3)

    means_file_4 = '../../data/training/8.csv'
    means4 = pd.read_csv(means_file_4)
    means = means.append(means4)

    means_file_5 = '../../data/training/9.csv'
    means5 = pd.read_csv(means_file_5)
    means = means.append(means5)

    # means_file_6 = '../data/training/1-remainder.csv'
    # means6 = pd.read_csv(means_file_6)
    # means = means.append(means6)

    training_set = means

    # file_all = './data/training/curated/all.csv'
    # training_set = pd.read_csv(file_all)

    return training_set


def get_training_data(with_origins=False, hp=False):
    subjects = ['subject01', 'subject02', 'subject03', 'subject04']
    training_data = pd.DataFrame()
    for subject in subjects:
        land_mark_file = 'data/{}/landmarks.csv'.format(subject)
        land_marks: pd.DataFrame = pd.read_csv(land_mark_file)
        land_marks = land_marks[(land_marks.correct != 0) & (land_marks.use != 0)]
        training_data = training_data.append(land_marks)

    if not with_origins:
        training_data.drop(['subject', 'image', 'correct', 'use'], axis=1, inplace=True, errors='ignore')
    # training_data = training_data[(training_data.sign != 27) & (training_data.sign != 17) & (training_data.sign != 30)]
    if not hp:
        training_data = training_data[
            (training_data.sign != 7) & (training_data.sign != 17) & (training_data.sign != 30)]

    return training_data


def _get_validation_set():
    means_file = '../../data/training/reference-signs-aw-01-right.csv'
    means: pd.DataFrame = pd.read_csv(means_file)

    means_file_2 = '../../data/training/reference-signs-aw-01-left.csv'
    means2 = pd.read_csv(means_file_2)
    means = means.append(means2)

    # means_file_3 = './data/training/reference-signs-geshani.csv'
    # means3 = pd.read_csv(means_file_3)
    # means = means.append(means3)

    means_file_4 = '../../data/training/8.csv'
    means4 = pd.read_csv(means_file_4)
    means = means.append(means4)

    means_file_5 = '../../data/training/9.csv'
    means5 = pd.read_csv(means_file_5)
    means = means.append(means5)

    file_all = '../../data/training/curated/all.csv'
    training_set = pd.read_csv(file_all)

    diffs = pd.concat([means, training_set]).drop_duplicates(keep=False)
    return diffs


def cascaded_classifier_worker(processed_q, training_data, video=False):
    classifier = CascadedClassifier(training_data)
    previous_sign = None
    filter = OutputFilter()
    while True:
        try:
            if not processed_q.empty():
                vertices, angles = processed_q.get()
                candidate_signs = classifier.classify_cascaded(vertices, angles)
                if not video:
                    if candidate_signs and candidate_signs[0]['class'] != previous_sign:
                        logging.info('Classification result {}'.format(candidate_signs))
                        previous_sign = candidate_signs[0]['class']
                else:
                    output = filter.filter(candidate_signs)
                    logging.info(output)
            else:
                pass
        except Exception as e:
            logging.error(e)
            break


def classifier_worker(processed_q, means, method):
    logging.info('Classifier worker running. Method: {}'.format(method))
    if method == ClassificationMethods.FLAT_COORDINATES:
        classifier = ClassifierByFlatCoordinates(means, [])
    if method == ClassificationMethods.ANGLES:
        classifier = ClassifierByAngles(means)
    if method == ClassificationMethods.ANGLES_AND_FLAT_CO:
        classifier = ClassifierByAnglesAndCoordinates(means)
    if method == ClassificationMethods.ENSEMBLE_1:
        classifier = CascadedClassifier(means)
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


def is_wrong_estimate(data_set_id, sign, wrong_estimates_df):
    filtered = wrong_estimates_df[(wrong_estimates_df.sign == sign) & (wrong_estimates_df.data_set == data_set_id)]
    return not filtered.empty


def _evaluate_image_data(labels, classifier, base_path):
    results_for_file = []
    for index, row in labels.iterrows():
        if row['label'] == 50: continue
        # if row['label'] != 21: continue
        # image = cv2.imread("./data/images/{}/{}".format(file_id, row["file_name"]))
        land_marks = mp_estimate_pose_from_image(base_path + "/images/{}".format(row["file_name"]))
        if not land_marks: continue
        land_marks, angles = pre_process_single_frame(land_marks)
        prediction = classifier.classify_cascaded(land_marks, angles)
        # if prediction[0].get('class') == 'NA':
        #     continue
        # prediction = rule_based_classify(prediction[0], angles)
        prediction[0].update({'truth_sign': LABEL_VS_INDEX.get(row['label'])})
        # result_row = pd.DataFrame(prediction[0], index=[0])
        results_for_file.append(prediction[0])
    results_df_for_file = pd.DataFrame(results_for_file)
    return results_df_for_file


def _get_mode_sign(video, start_frame_no, end_frame_no, truth_sign, classifier):
    all_detected_signs = []
    tot_duration_est = 0
    tot_frames_est = 0
    tot_duration_clf = 0
    tot_frames_clf = 0
    for frame in range(round(start_frame_no), round(end_frame_no), 2):
        image = get_static_frame2(video, frame)
        start = tm_mod.process_time()
        land_marks = mp_estimate_pose_static(image)
        end = tm_mod.process_time()
        duration = end - start
        tot_duration_est = tot_duration_est + duration
        tot_frames_est = tot_frames_est + 1
        if not land_marks: continue
        start = tm_mod.process_time()
        land_marks, angles = pre_process_single_frame(land_marks)
        # render_static(land_marks)
        prediction = classifier.classify_cascaded(land_marks, angles)
        # if prediction[0].get('class') == 'NA':
        #     continue
        # prediction = rule_based_classify(prediction[0], angles)
        end = tm_mod.process_time()
        duration = end - start
        tot_duration_clf = tot_duration_clf + duration
        tot_frames_clf = tot_frames_clf + 1
        prediction[0].update({'truth_sign': LABEL_VS_INDEX.get(truth_sign)})
        all_detected_signs.extend(prediction)
    mode_sign = statistics.mode([pred.get('class') for pred in all_detected_signs]) if len(all_detected_signs) else None
    # print(tot_duration_est, tot_frames_est, tot_frames_est/tot_duration_est)
    # print(tot_duration_clf, tot_frames_clf, tot_frames_clf/tot_duration_clf)
    if mode_sign:
        return {'class': mode_sign, 'truth_sign': LABEL_VS_INDEX.get(truth_sign)}


def _evaluate_for_selected_frames(start_frame, total_frames, classifier, truth_sign, video):
    results_list_for_file = []
    # for frame in [start_frame + total_frames * .1 * i for i in range(1, 10)]:
    for frame in [start_frame + total_frames * .25 * i for i in [1, 3]]:
        image = get_static_frame2(video, frame)
        land_marks = mp_estimate_pose_static(image)
        if not land_marks: continue
        land_marks, angles = pre_process_single_frame(land_marks)
        # render_static(land_marks)
        prediction = classifier.classify(land_marks)
        if prediction[0].get('class') == 'NA':
            continue
        prediction = rule_based_classify(prediction[0], angles)
        prediction[0].update({'truth_sign': LABEL_VS_INDEX.get(truth_sign)})
        results_list_for_file.extend(prediction)
    return results_list_for_file


def _evaluate_all_frames(start_frame, end_frame, total_frames, truth_sign, classifier, video):
    results_list_for_file = []
    start1 = start_frame + total_frames * .1
    end1 = start_frame + total_frames / 2
    prediction1 = _get_mode_sign(video, start1, end1, truth_sign, classifier)
    if prediction1: results_list_for_file.append(prediction1)

    start2 = start_frame + total_frames / 2 + 1
    end2 = end_frame - total_frames * .1
    prediction2 = _get_mode_sign(video, start2, end2, truth_sign, classifier)
    if prediction2: results_list_for_file.append(prediction2)
    return results_list_for_file


def _evaluate_video_data(labels, classifier, path, data_set_id):
    video = path
    results_list_for_file = []
    we_file_path = '../data/incorrect_estimates.csv'
    wrong_estimates_df = pd.read_csv(we_file_path)
    for index, row in labels.iterrows():
        truth_sign = row['label']
        if is_wrong_estimate(data_set_id, truth_sign, wrong_estimates_df): continue
        if truth_sign == 50: continue
        # if truth_sign != 21: continue
        start_frame = row['start']
        end_frame = row['end']

        total_frames = end_frame - start_frame

        # results = _evaluate_for_selected_frames(start_frame, total_frames, classifier, truth_sign, video)

        results = _evaluate_all_frames(start_frame, end_frame, total_frames, truth_sign, classifier, video)
        results_list_for_file.extend(results)

    # if len(results_list_for_file) == 0: continue
    # preds = [x['class'] for x in results_list_for_file]
    # mode_pred = statistics.mode(preds)
    # mode_pred = [{'class': mode_pred, 'truth_sign': LABEL_VS_INDEX.get(row['label'])}]
    # result_row = pd.DataFrame(mode_pred, index=[0])
    results_for_file = pd.DataFrame(results_list_for_file)
    return results_for_file


def validate():
    all_results = pd.DataFrame()
    training_data = get_training_data()
    # classifier = EnsembleClassifier(training_data, None)
    classifier = CascadedClassifier(training_data)
    for subject_id in [5, 6, 7, 8, 9]:
        base_path = '../data/subject{}'.format(subject_id if subject_id > 9 else '0{}'.format(subject_id))
        file_path = base_path + '/labels.csv'
        labels: pd.DataFrame = pd.read_csv(file_path)

        meta_data_fpath = base_path + '/meta.json'
        with open(meta_data_fpath) as f_in:
            meta_data = json.load(f_in)
        # labels = labels[labels['label'] == 1]
        # video_m = video_meta.get(subject_id)
        file_type = meta_data.get("type")
        if file_type == 'VIDEO':
            video_path = base_path + '/video.{}'.format(meta_data.get("format"))
            results_for_file = _evaluate_video_data(labels, classifier, video_path, subject_id)
        else:
            results_for_file = _evaluate_image_data(labels, classifier, base_path)

        if results_for_file.empty: continue
        all_results = all_results.append(results_for_file)

        acc = accuracy_score(results_for_file['truth_sign'], results_for_file['class'])
        pre = precision_score(results_for_file['truth_sign'], results_for_file['class'], average='macro')
        logging.info('Results for the file {}\nacc {} \npre {}'.format(subject_id, acc, pre))

    acc = accuracy_score(all_results['truth_sign'], all_results['class'])
    pre = precision_score(all_results['truth_sign'], all_results['class'], average='macro')

    all_results.groupby('truth_sign').apply(lambda x: accuracy_score(x['truth_sign'], x['class']))
    all_results.groupby('truth_sign').apply(lambda x: precision_score(x['truth_sign'], x['class'], average='macro'))

    logging.info('Results for all \nacc {} \npre {}'.format(acc, pre))
    plot_cnf_matrix(all_results)


def find_hyper_parameters(model: str):
    training_data = get_training_data()
    if model == 'LR':
        LogisticRegressionPoseClassifier(training_data).tune_hpp()
    if model == 'KNN':
        KNNPoseClassifier(training_data).tune_hpp()
    if model == 'RF':
        RandomForestPoseClassifier(training_data).tune_hpp()
    if model == 'XG':
        XGBoostPoseClassifier(training_data).tune_hpp()
    if model == 'DT':
        DesicionTreePoseClassifier(training_data).tune_hpp()
    if model == 'NB':
        NaiveBaysPoseClassifier(training_data).tune_hpp()
    if model == 'EN':
        CascadedClassifier(training_data).hpp_level1_ensemble(training_data)
    if model == 'CAS':
        training_data = get_training_data(hp=True)
        CascadedClassifier(training_data).hpp_cascaded(training_data)
    if model == 'OP':
        # training_data = get_training_data(hp=True)
        tune_hpp_classifier_option(training_data)
    if model == 'AN':
        CascadedClassifierAngles(training_data).hpp_level1_ensemble(training_data)


def lc(model: str):
    training_data = get_training_data()
    if model == 'LR':
        LogisticRegressionPoseClassifier(training_data).plot_lc()
    if model == 'KNN':
        KNNPoseClassifier(training_data).plot_lc()
    if model == 'RF':
        RandomForestPoseClassifier(training_data).plot_lc()
    if model == 'XG':
        XGBoostPoseClassifier(training_data).tune_hpp()
    if model == 'DT':
        DesicionTreePoseClassifier(training_data).tune_hpp()
    if model == 'NB':
        NaiveBaysPoseClassifier(training_data).tune_hpp()


def get_measures():
    df = pd.read_csv('all_results')
    clf_r = classification_report(df['truth_sign'], df['class'])
    logging.info(clf_r)


if __name__ == '__main__':
    logging.info('Initiating classification...')
    video_m = video_meta.get(4)
    video = video_m.get('location')
    # fps = video_m.get('fps')
    #
    # time = 31.2

    # process_single_frame(video, time, fps, classify=True, method=ClassificationMethods.ANGLES)

    # process_video(video, classify=True, method=ClassificationMethods.ENSEMBLE_1)
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

    # validate()
    # find_hyper_parameters()
    # _get_training_data()
    # _temp_find_frame()
    get_measures()
