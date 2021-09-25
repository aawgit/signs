import logging
import multiprocessing
import csv
import threading

import pandas as pd
from scipy.spatial import distance
import numpy as np

from feature_extraction.pre_processor import pre_process, get_angles
from feature_extraction.renderer import render
from feature_extraction.points_generator import dynamic_images
from utils.constants import LABEL_ORDER_CHAMINDA

logging.getLogger().setLevel(logging.INFO)
import pygame

from tkinter import *


def pose_estimation_worker(que):
    dynamic_images(que)


def pre_process_worker(pose_q, processed_q_1, processed_q_2):
    while True:
        try:
            if not pose_q.empty():
                current_vertices, frame_no = pose_q.get()
                current_vertices = pre_process(current_vertices)
                processed_q_1.put(current_vertices)
                processed_q_2.put((current_vertices, frame_no))
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


def classifier_worker(processed_q):
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


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=3)
    m = multiprocessing.Manager()
    hand_pose_q = m.Queue()
    workers_1 = pool.apply_async(pose_estimation_worker, (hand_pose_q,))

    pre_processed_q_1 = m.Queue()
    pre_processed_q_2 = m.Queue()
    workers_2 = pool.apply_async(pre_process_worker, (hand_pose_q, pre_processed_q_1, pre_processed_q_2))
    # workers_3 = pool.apply_async(labeller_worker, (pre_processed_q_2,))
    workers_3 = pool.apply_async(classifier_worker, (pre_processed_q_2,))

    render(pre_processed_q_1)
