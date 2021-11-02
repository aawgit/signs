import logging
from itertools import chain

import pandas as pd
import pygame
from scipy.spatial import distance

from feature_extraction.pre_processor import get_angles, un_flatten_points, get_angle_v2
from utils.constants import LABEL_ORDER_CHAMINDA, ClassificationMethods, JOINTS_FOR_ANGLES
from pose_estimation.vertices_mapper import EDGES_MEDIA_PIPE


def classifier_by_angles_worker(processed_q):
    logging.info('Classifier worker running...')

    means = pd.read_csv('./data/training/means.csv')[['thumb_angle', 'index_angle', 'middle_angle',
                                                      'ring_angle', 'pinky_angle', 'label']]

    def calculate_distance(th, ind, mi, ri, pi, angles):
        return distance.euclidean([th, ind, mi, ri, pi], angles)

    previous_sign = None
    while True:
        try:
            if not processed_q.empty():
                vertices, frame_no = processed_q.get()
                angles = get_angles(vertices)
                means['distance'] = means.apply(lambda x: calculate_distance(x.thumb_angle, x.index_angle,
                                                                             x.middle_angle, x.ring_angle,
                                                                             x.pinky_angle, angles), axis=1)
                matching_signs = means[means['distance'] < 20]
                if not matching_signs.empty:
                    index_of_sign_with_min_distance = \
                        matching_signs[matching_signs['distance'] == matching_signs['distance'].min()]['label'].item()
                    sign = LABEL_ORDER_CHAMINDA.get(index_of_sign_with_min_distance)
                    if sign != previous_sign:
                        logging.info('Sign is {}'.format(sign))
                        previous_sign = sign
            else:
                pass
        except Exception as e:
            logging.error(e)
            break

def classify_by_angles(landmark, mean_angles_df):

    angles = []

    for joint in JOINTS_FOR_ANGLES:
        limb2 = [landmark[joint[2]][i] - landmark[joint[1]][i] for i in range(0, 3)]
        limb1 = [landmark[joint[1]][i] - landmark[joint[0]][i] for i in range(0, 3)]
        angle = get_angle_v2(limb2, limb1)
        angles.append(angle)

    mean_angles_df['distance'] = mean_angles_df.drop(['sign','distance'], errors='ignore', axis=1) \
        .apply(lambda x: distance.euclidean(x, angles), axis=1)
    matching_signs = mean_angles_df[mean_angles_df['distance'] < 100]
    if not matching_signs.empty:
        index_of_sign_with_min_distance = \
            matching_signs[matching_signs['distance'] == matching_signs['distance'].min()]['sign'].item()
        sign = LABEL_ORDER_CHAMINDA.get(index_of_sign_with_min_distance)
        return sign

def classify_by_flat_coordinates(flattened_coordinates, flattened_mean_coordinates):
    flattened_coordinates = list(flattened_coordinates)
    del flattened_coordinates[5]
    del flattened_coordinates[8]
    del flattened_coordinates[11]
    del flattened_coordinates[14]
    flattened_coordinates = list(chain(*flattened_coordinates[1:]))
    means_with_selected_cols = flattened_mean_coordinates.drop([
        'sign',
        'distance',
        '0_0', '0_1', '0_2',
        '5_0', '5_1', '5_2',
        '9_0', '9_1', '9_2',
        '13_0', '13_1', '13_2',
        '17_0', '17_1', '17_2',
    ], errors='ignore', axis=1)
    flattened_mean_coordinates['distance'] = means_with_selected_cols \
        .apply(lambda x: distance.euclidean(x, flattened_coordinates), axis=1)
    matching_signs = flattened_mean_coordinates[flattened_mean_coordinates['distance'] < 2]
    if not matching_signs.empty:
        index_of_sign_with_min_distance = \
            matching_signs[matching_signs['distance'] == matching_signs['distance'].min()]['sign'].item()
        sign = LABEL_ORDER_CHAMINDA.get(index_of_sign_with_min_distance)
        return sign

def get_mean_angles(flattened_mean_coordinates):
    signs = flattened_mean_coordinates['sign']
    means_list = flattened_mean_coordinates.drop('sign', axis=1).values.tolist()
    mean_angles_df_cols = [str(point_pair) for point_pair in JOINTS_FOR_ANGLES]
    # mean_angles_df_cols.append('sign')
    mean_angles_df = pd.DataFrame(columns=mean_angles_df_cols)

    for row in means_list:
        angles_for_the_row = []
        row = un_flatten_points(row)
        for joint in JOINTS_FOR_ANGLES:
            limb2 = [row[joint[2]][i] - row[joint[1]][i] for i in range(0, 3)]
            limb1 = [row[joint[1]][i] - row[joint[0]][i] for i in range(0, 3)]
            reference_angle = get_angle_v2(limb2, limb1)
            angles_for_the_row.append(reference_angle)
        row_angles_df = pd.DataFrame([angles_for_the_row], columns=mean_angles_df_cols)
        mean_angles_df = mean_angles_df.append(row_angles_df)
    mean_angles_df = pd.concat((mean_angles_df.reset_index(drop=True), signs.rename('sign')), axis=1)
    return mean_angles_df


def classifier_worker(processed_q, means, method):
    logging.info('Classifier worker running. Method: {}'.format(method))
    if method == ClassificationMethods.ANGLES:
        means = get_mean_angles(means)
    previous_sign = None
    while True:
        try:
            if not processed_q.empty():
                vertices = processed_q.get()
                if method==ClassificationMethods.FLAT_COORDINATES:
                    sign = classify_by_flat_coordinates(vertices, means)
                elif method == ClassificationMethods.ANGLES:
                    sign = classify_by_angles(vertices, means)
                if sign and sign != previous_sign:
                    logging.info('Sign is {}'.format(sign))
                    previous_sign = sign
            else:
                pass
        except Exception as e:
            logging.error(e)
            break


class KeyInputHolder:
    def __init__(self):
        self.count = 0
        self.is_label = False

    def mark_sign(self):
        self.count = self.count + 1
        self.is_label = True
        logging.info('Current sign {} '.format(LABEL_ORDER_CHAMINDA.get(self.count)))

    def clear_sign(self):
        self.is_label = False

    def get_current_label(self):
        if self.is_label: return self.count

    def get_is_label(self):
        return self.is_label

    def set_dummy_window_for_key_press(self):
        pygame.init()
        BLACK = (0, 0, 0)
        WIDTH = 100
        HEIGHT = 100
        windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

        windowSurface.fill(BLACK)
