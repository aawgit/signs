import csv
import logging
from itertools import chain

import pandas as pd
import pygame
from scipy.spatial import distance

from feature_extraction.pre_processor import get_angles
from utils.constants import LABEL_ORDER_CHAMINDA


def labeller_worker(processed_q):
    logging.info('Labeller worker running...')

    key_input_holder = KeyInputHolder()
    key_input_holder.set_dummy_window_for_key_press()

    with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "thumb_angle", "index_angle", "middle_angle", "ring_angle", "pinky_angle", "label"])

        while True:
            try:
                if not processed_q.empty():
                    vertices, frame_no = processed_q.get()
                    angles = get_angles(vertices)
                    label = None

                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                            key_input_holder.mark_sign()

                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                            key_input_holder.clear_sign()

                    if key_input_holder.get_is_label(): label = key_input_holder.get_current_label()
                    writer.writerow([frame_no, *angles, label])
                else:
                    pass
            except Exception as e:
                logging.error(e)
                break


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


def classifier_by_vertices_worker(processed_q):
    logging.info('Classifier worker running...')

    means = pd.read_csv('./data/training/vertices_means_chaminda.csv')

    previous_sign = None
    while True:
        try:
            if not processed_q.empty():
                vertices, frame_no = processed_q.get()
                vertices = list(vertices)
                del vertices[5]
                del vertices[8]
                del vertices[11]
                del vertices[14]
                vertices = list(chain(*vertices[1:]))
                means = means[means.sign != 32]
                means['distance'] = means.drop([
                    'sign',
                    'distance',
                    '5_0', '5_1', '5_2',
                    '9_0', '9_1', '9_2',
                    '13_0', '13_1', '13_2',
                    '17_0', '17_1', '17_2',],errors='ignore', axis=1)\
                    .apply(lambda x: distance.euclidean(x, vertices), axis=1)
                matching_signs = means[means['distance'] < 2]
                if not matching_signs.empty:
                    index_of_sign_with_min_distance = \
                        matching_signs[matching_signs['distance'] == matching_signs['distance'].min()]['sign'].item()
                    sign = LABEL_ORDER_CHAMINDA.get(index_of_sign_with_min_distance)
                    if sign != previous_sign:
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