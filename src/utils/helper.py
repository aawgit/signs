import logging

import pygame
from scipy.stats import mode

from src.feature_extraction.pre_processor import flatten_points
from src.utils.constants import LABEL_VS_INDEX


class KeyInputHolder:
    def __init__(self):
        self.count = 0
        self.is_label = False

    def mark_sign(self):
        self.count = self.count + 1
        self.is_label = True
        logging.info('Current sign {} '.format(LABEL_VS_INDEX.get(self.count)))

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


class OutputFilter:
    def __init__(self, buffer_size=30):
        self.buffer = ['No sign']*buffer_size

    def filter(self, candidate_signs):
        sign = candidate_signs[0]['class']
        self.buffer.pop(0)
        self.buffer.append(sign)
        p = mode(self.buffer).mode[0]
        return p


def _get_landmark_col_names():
    col_names = flatten_points([('{}_0'.format(i), '{}_1'.format(i), '{}_2'.format(i)) for i in range(0, 21)])
    return col_names