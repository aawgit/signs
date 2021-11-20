import logging
from itertools import chain

import pandas as pd
import pygame
from scipy.spatial import distance

from feature_extraction.pre_processor import un_flatten_points, get_angle_v2, normalize_flat_coordinates_scale
from utils.constants import LABEL_VS_INDEX, JOINTS_FOR_ANGLES


class NearestNeighborPoseClassifier:
    def __init__(self, cluster_means: pd.DataFrame, threshold):
        self.cluster_means = self.unify_cluster_mean_features(cluster_means)
        self.threshold = threshold

    def unify_cluster_mean_features(self, cluster_means):
        pass

    def unify_frame_features(self, landmark):
        pass

    def classify(self, landmark):
        incoming_frame = self.unify_frame_features(landmark)
        # logging.info(incoming_frame)
        self.cluster_means['distance'] = self.cluster_means.drop(['sign', 'distance'], errors='ignore', axis=1) \
            .apply(lambda x: distance.euclidean(x, incoming_frame), axis=1)
        candidates_df = self.cluster_means.nsmallest(5, 'distance').sort_values(by=['distance'])[self.cluster_means.distance<2.5]
        candidates_df['class'] = candidates_df['sign'].apply(lambda x: LABEL_VS_INDEX.get(x))

        joint_wise_deviation = (candidates_df.drop(['sign', 'distance', 'class'], axis=1) - incoming_frame).abs()
        max_deviations = joint_wise_deviation.apply('max', axis=1)
        alternative = candidates_df.loc[max_deviations.idxmin()]['class']
        logging.info('Alternative: {}'.format(alternative))
        #
        # lowest = max_deviations.iloc[0]
        # for deviation in max_deviations:
        #     if deviation<lowest:
        #         lowest = deviation
        #         break
        #     lowest = deviation
        #
        # return candidates_df[max_deviations == lowest][['class', 'distance']].to_dict('records')


        return candidates_df[['class', 'distance']].to_dict('records')
        # TODO: Re-evaluate the necessity for limiting the distance
        # matching_signs = self.cluster_means[self.cluster_means['distance'] < self.threshold]
        # if not matching_signs.empty:
        #     index_of_sign_with_min_distance = \
        #         matching_signs[matching_signs['distance'] == matching_signs['distance'].min()]['sign'].item()
        #     sign = LABEL_ORDER_CHAMINDA.get(index_of_sign_with_min_distance)
        #     return sign


class ClassifierByFlatCoordinates(NearestNeighborPoseClassifier):
    def __init__(self, cluster_means: pd.DataFrame, threshold, vertices_to_ignore=None):
        if vertices_to_ignore is None:
            vertices_to_ignore = [0, 5, 9, 13, 17]
        self.vertices_to_ignore = vertices_to_ignore
        super().__init__(cluster_means, threshold)

    def unify_cluster_mean_features(self, cluster_means: pd.DataFrame):
        cluster_means = normalize_flat_coordinates_scale(cluster_means)
        if self.vertices_to_ignore:
            cols_to_drop = []
            for vertex in self.vertices_to_ignore:
                for i in range(0, 3):
                    cols_to_drop.append('{}_{}'.format(vertex, i))

            unified_cluster_means = cluster_means.drop(cols_to_drop, errors='ignore', axis=1)
        else:
            unified_cluster_means = cluster_means
        return unified_cluster_means

    def unify_frame_features(self, landmark):
        landmark = list(landmark)
        if self.vertices_to_ignore:
            for idx, vertex in enumerate(self.vertices_to_ignore):
                del landmark[vertex - idx]
        flattened_coordinates = list(chain(*landmark))
        return flattened_coordinates


class ClassifierByAngles(NearestNeighborPoseClassifier):
    def __init__(self, cluster_means: pd.DataFrame, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        super().__init__(cluster_means, 2)

    def unify_cluster_mean_features(self, cluster_means):
        signs = cluster_means['sign']
        means_list = cluster_means.drop('sign', axis=1).values.tolist()
        mean_angles_df_cols = [str(point_pair) for point_pair in JOINTS_FOR_ANGLES]
        mean_angles_df = pd.DataFrame(columns=mean_angles_df_cols)

        for row in means_list:
            angles_for_the_row = []
            row = un_flatten_points(row)
            for joint in JOINTS_FOR_ANGLES:
                limb2 = [row[joint[2]][i] - row[joint[1]][i] for i in range(0, 3)]
                limb1 = [row[joint[1]][i] - row[joint[0]][i] for i in range(0, 3)]
                reference_angle = get_angle_v2(limb2, limb1)
                angles_for_the_row.append(reference_angle / 90)
            row_angles_df = pd.DataFrame([angles_for_the_row], columns=mean_angles_df_cols)
            mean_angles_df = mean_angles_df.append(row_angles_df)
        mean_angles_df = pd.concat((mean_angles_df.reset_index(drop=True), signs.rename('sign')), axis=1)
        return mean_angles_df

    def unify_frame_features(self, landmark):
        angles = []
        for joint in JOINTS_FOR_ANGLES:
            limb2 = [landmark[joint[2]][i] - landmark[joint[1]][i] for i in range(0, 3)]
            limb1 = [landmark[joint[1]][i] - landmark[joint[0]][i] for i in range(0, 3)]
            angle = get_angle_v2(limb2, limb1)
            angles.append(angle / 90)

        return angles


class ClassifierByAnglesAndCoordinates(NearestNeighborPoseClassifier):
    def __init__(self, cluster_means: pd.DataFrame, threshold, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        self.coordinateClassifier = ClassifierByFlatCoordinates(cluster_means, threshold, vertices_to_ignore)
        self.angleClassifier = ClassifierByAngles(cluster_means, threshold)
        super().__init__(cluster_means, 2)

    def unify_cluster_mean_features(self, cluster_means):
        unified_coordinates = self.coordinateClassifier.cluster_means
        unified_angles = self.angleClassifier.cluster_means
        unified_co_and_angle = pd.merge(unified_coordinates, unified_angles, on='sign')
        return unified_co_and_angle

    def unify_frame_features(self, landmark):
        unified_coordinates = self.coordinateClassifier.unify_frame_features(landmark)
        unified_angles = self.angleClassifier.unify_frame_features(landmark)
        unified_coordinates.extend(unified_angles)
        return unified_coordinates


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


