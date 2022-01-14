import logging
from itertools import chain

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import mode

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import pygame
from scipy.spatial import distance
from sklearn import tree

from feature_extraction.pre_processor import un_flatten_points, get_angle_v2, normalize_flat_coordinates_scale, \
    flatten_points
from utils.constants import LABEL_VS_INDEX, EDGE_PAIRS_FOR_ANGLES


class NearestNeighborPoseClassifier:
    def __init__(self, cluster_means: pd.DataFrame, threshold):
        self.cluster_means = self.unify_cluster_mean_features(cluster_means)
        self.threshold = threshold

    def unify_cluster_mean_features(self, cluster_means):
        pass

    def unify_frame_features(self, landmark):
        pass

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.unify_frame_features(landmark)
        # logging.info(incoming_frame)
        self.cluster_means['distance'] = self.cluster_means.drop(['sign', 'distance'], errors='ignore', axis=1) \
            .apply(lambda x: distance.euclidean(x, incoming_frame), axis=1)
        candidates_df = self.cluster_means.nsmallest(5, 'distance').sort_values(by=['distance'])
        candidates_df['class'] = candidates_df['sign'].apply(lambda x: LABEL_VS_INDEX.get(x))

        # joint_wise_deviation = (candidates_df.drop(['sign', 'distance', 'class'], axis=1) - incoming_frame).abs()
        # max_deviations = joint_wise_deviation.apply('max', axis=1)
        # alternative = candidates_df.loc[max_deviations.idxmin()]['class']
        # logging.info('Alternative: {}'.format(alternative))
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
        # cluster_means = normalize_flat_coordinates_scale(cluster_means)
        if self.vertices_to_ignore:
            cols_to_drop = []
            for vertex in self.vertices_to_ignore:
                for i in range(0, 3):
                    cols_to_drop.append('{}_{}'.format(vertex, i))

            unified_cluster_means = cluster_means.drop(cols_to_drop, errors='ignore', axis=1)
        else:
            unified_cluster_means = cluster_means
        # unified_cluster_means = self._drop_z_axis(unified_cluster_means)
        return unified_cluster_means.reset_index(drop=True)

    def _drop_z_axis(self, cluster_means):
        cols_to_drop = [col for col in cluster_means.columns.values if col.endswith('2')]

        remaining_cluster_means = cluster_means.drop(cols_to_drop, errors='ignore', axis=1)
        return remaining_cluster_means

    def unify_frame_features(self, landmark):
        landmark = list(landmark)
        if self.vertices_to_ignore:
            for idx, vertex in enumerate(self.vertices_to_ignore):
                del landmark[vertex - idx]
        flattened_coordinates = list(chain(*landmark))
        # flattened_coordinates = self.remve_z_coordinate(flattened_coordinates)
        return flattened_coordinates

    def remve_z_coordinate(self, flattened_coordinates):
        k = 3
        del flattened_coordinates[k-1::k]
        return flattened_coordinates

class ClassifierByAngles(NearestNeighborPoseClassifier):
    def __init__(self, cluster_means: pd.DataFrame, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        super().__init__(cluster_means, 2)

    def unify_cluster_mean_features(self, cluster_means):
        signs = cluster_means['sign']
        means_list = cluster_means.drop(['sign', 'source'], axis=1, errors='ignore').values.tolist()
        mean_angles_df_cols = [str(point_pair) for point_pair in EDGE_PAIRS_FOR_ANGLES]
        mean_angles_df = pd.DataFrame(columns=mean_angles_df_cols)

        for row in means_list:
            angles_for_the_row = []
            row = un_flatten_points(row)
            for limb_pair in EDGE_PAIRS_FOR_ANGLES:
                limb2 = [row[limb_pair[1][1]][i] - row[limb_pair[1][0]][i] for i in range(0, 3)]
                limb1 = [row[limb_pair[0][1]][i] - row[limb_pair[0][0]][i] for i in range(0, 3)]
                reference_angle = get_angle_v2(limb2, limb1)
                angles_for_the_row.append((reference_angle - 90) / 90)
            row_angles_df = pd.DataFrame([angles_for_the_row], columns=mean_angles_df_cols)
            mean_angles_df = mean_angles_df.append(row_angles_df)
        mean_angles_df = pd.concat((mean_angles_df.reset_index(drop=True), signs.rename('sign').reset_index(drop=True)),
                                   axis=1).drop('index', axis=1, errors='ignore')
        return mean_angles_df

    def unify_frame_features(self, landmark):
        angles = []
        for limb_pair in EDGE_PAIRS_FOR_ANGLES:
            limb2 = [landmark[limb_pair[1][1]][i] - landmark[limb_pair[1][0]][i] for i in range(0, 3)]
            limb1 = [landmark[limb_pair[0][1]][i] - landmark[limb_pair[0][0]][i] for i in range(0, 3)]
            angle = get_angle_v2(limb2, limb1)
            angles.append((angle - 90) / 90)

        return angles


class ClassifierByAnglesAndCoordinates(NearestNeighborPoseClassifier):
    def __init__(self, cluster_means: pd.DataFrame, threshold, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        self.coordinateClassifier = ClassifierByFlatCoordinates(cluster_means, threshold, vertices_to_ignore)
        self.angleClassifier = ClassifierByAngles(cluster_means, threshold)
        super().__init__(cluster_means, 2)

    def unify_cluster_mean_features(self, cluster_means):
        unified_coordinates = self.coordinateClassifier.cluster_means.drop(['sign', 'source'], axis=1, errors='ignore')
        unified_angles = self.angleClassifier.cluster_means
        unified_co_and_angle = pd.concat([unified_coordinates, unified_angles], axis=1)
        return unified_co_and_angle

    def unify_frame_features(self, landmark):
        unified_coordinates = self.coordinateClassifier.unify_frame_features(landmark)
        unified_angles = self.angleClassifier.unify_frame_features(landmark)
        unified_coordinates.extend(unified_angles)
        return unified_coordinates


class DecisionTreeClassifier(ClassifierByAnglesAndCoordinates):
    def __init__(self, cluster_means: pd.DataFrame, threshold, vertices_to_ignore=None):
        super().__init__(cluster_means, threshold)
        self.X = self.cluster_means.drop('sign', axis=1)
        self.Y = self.cluster_means['sign']
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(self.X, self.Y)

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.unify_frame_features(landmark)
        self.clf.predict([incoming_frame])


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


class ExperimentalClassifier(ClassifierByAnglesAndCoordinates):
    def __init__(self, cluster_means: pd.DataFrame, threshold):
        super().__init__(cluster_means, threshold)
        # self.cluster_means = self.get_only_important_features_training(self.cluster_means)
        X_train = self.cluster_means.drop('sign', axis=1).values
        y_train = self.cluster_means['sign'].values
        self.lr = LogisticRegression(random_state=0).fit(X_train, y_train)  # - Good
        self.knn1 = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=4, weights='distance')  # - Good
        # euc: 0.5689, 0.5027
        # manhattan: 0.5377, 0.4836
        # chebyshev: 0.5114, 0.4996
        # minkowski: 0.5689, 0.5027; p=3 0.5340, 0.4710; p=4 0.5737, 0.4908, p=4 weights=distance 0.5832, 0.5507
        # wminkowski:
        # seuclidean:
        # mahalanobis:
        self.rf1 = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini')
        # self.rf2 = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')
        # - Good - 0.58 acc; random state not set
        # gini 0.5605, 0.54747 rs 0;
        # entropy 0.5473, 0.5303
        # self.gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        #                                      max_depth=1, random_state=0)
        self.lr.fit(X_train, y_train)
        self.knn1.fit(X_train, y_train)
        self.rf1.fit(X_train, y_train)
        # self.clf4.fit(self.cluster_means.drop('sign', axis=1), self.cluster_means['sign'])

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.unify_frame_features(landmark)
        # incoming_frame = self.get_only_important_features(incoming_frame)

        pred_knn = self.knn1.predict([incoming_frame, ])
        prob = self.knn1.predict_proba([incoming_frame, ])
        prob = max(*prob)
        if prob > 0.65:
            p = pred_knn[0]
        else:
            # finger_wise_prediction = classify_finger_wise(self.cluster_means, incoming_frame)
            pred_lr = self.lr.predict([incoming_frame, ])
            pred_rf1 = self.rf1.predict([incoming_frame, ])
            # p4 = self.clf4.predict([incoming_frame, ])
            p = mode([pred_rf1[0], pred_lr[0], pred_knn[0]]).mode[0]
        # if p:
        #     if False:#p.count>1:
        #         p = p[0]
        #         prediction = [{'class': LABEL_VS_INDEX.get(p[0])}]
        #     else:
        #         prob = self.clf1.predict_proba([incoming_frame, ])
        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction
        # else:
        #     return [{'class': 'NA', 'distance': 'NA'}]
        # if p4:
        #     prediction = [{'class': LABEL_VS_INDEX.get(p4[0])}]
        #     return prediction
        # else: return [{'class': 'NA', 'distance': 'NA'}]

    def classify2(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.unify_frame_features(landmark)
        # incoming_frame = self.get_only_important_features(incoming_frame)

        pred_knn = self.knn1.predict([incoming_frame, ])
        pred_lr = self.lr.predict([incoming_frame, ])
        pred_rf1 = self.rf1.predict([incoming_frame, ])

        p = mode([
            pred_rf1[0],
            pred_lr[0],
            pred_knn[0]
        ]).mode[0]

        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction

class EnsembleClassifier(NearestNeighborPoseClassifier):
    def __init__(self, cluster_means: pd.DataFrame, threshold, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        self.coordinateClassifier = ClassifierByFlatCoordinates(cluster_means, threshold, vertices_to_ignore)
        self.angleClassifier = ClassifierByAngles(cluster_means, threshold)

        super().__init__(cluster_means, threshold)

        X_train_c = self.coordinateClassifier.cluster_means.drop(['sign', 'source'], axis=1, errors='ignore').values
        y_train_c = self.coordinateClassifier.cluster_means['sign'].values

        self.lr_c = LogisticRegression(random_state=0).fit(X_train_c, y_train_c)  # - Good
        self.knn1_c = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=4, weights='distance')  # - Good
        self.rf1_c = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini')

        self.lr_c.fit(X_train_c, y_train_c)
        self.knn1_c.fit(X_train_c, y_train_c)
        self.rf1_c.fit(X_train_c, y_train_c)

        X_train_a = self.angleClassifier.cluster_means.drop('sign', axis=1).values
        y_train_a = self.angleClassifier.cluster_means['sign'].values

        self.lr_a = LogisticRegression(random_state=0).fit(X_train_a, y_train_a)  # - Good
        self.knn1_a = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=4, weights='distance')  # - Good
        self.rf1_a = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini')

        self.lr_a.fit(X_train_a, y_train_a)
        self.knn1_a.fit(X_train_a, y_train_a)
        self.rf1_a.fit(X_train_a, y_train_a)

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame_c = self.coordinateClassifier.unify_frame_features(landmark)

        pred_knn_c = self.knn1_c.predict([incoming_frame_c, ])
        pred_lr_c = self.lr_c.predict([incoming_frame_c, ])
        pred_rf1_c = self.rf1_c.predict([incoming_frame_c, ])

        incoming_frame_a = self.angleClassifier.unify_frame_features(landmark)
        pred_knn_a = self.knn1_a.predict([incoming_frame_a, ])
        pred_lr_a = self.lr_a.predict([incoming_frame_a, ])
        pred_rf1_a = self.rf1_a.predict([incoming_frame_a, ])

        p1 = mode([pred_rf1_c[0],
            pred_lr_c[0],
            pred_knn_c[0]]).mode[0]

        p2 = mode( [pred_rf1_a[0],
            pred_lr_a[0],
            pred_knn_a[0]]).mode[0]

        p = mode([p2, p1]).mode[0]

        # p = mode([
        #     pred_rf1_c[0],
        #     pred_lr_c[0],
        #     pred_knn_c[0],
        #     pred_rf1_a[0],
        #     pred_lr_a[0],
        #     pred_knn_a[0]
        # ]).mode[0]

        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction

important_feature_idx = [0, 3, 5, 6, 7, 8, 9, 10, 11, 13, 16, 19, 20, 22, 25, 28, 31, 34, 35, 37, 38, 40, 43,
                         44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]


def get_only_important_features(self, features: list):
    important_features = []
    for idx, val in enumerate(features):
        if idx in self.important_feature_idx:
            important_features.append(val)
    return important_features


def get_only_important_features_training(self, training_data: pd.DataFrame):
    return training_data.iloc[:, [*self.important_feature_idx, 63]]


def move_fingers_to_origin_training(self, training_data: pd.DataFrame):
    pass


class FingerwiseCompareClassifier(ClassifierByAnglesAndCoordinates):
    def __init__(self, cluster_means: pd.DataFrame, threshold):
        super().__init__(cluster_means, threshold)
        self.finger_joints = ((1, 2, 3, 4),
                              (6, 7, 8),
                              (10, 11, 12),
                              (14, 15, 16),
                              (18, 19, 20))

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.unify_frame_features(landmark)
        prediction = classify_finger_wise(incoming_frame, self.cluster_means)
        return prediction
        # candidates_df = self.cluster_means.nsmallest(5, 'distance').sort_values(by=['distance'])
        # candidates_df['class'] = candidates_df['sign'].apply(lambda x: LABEL_VS_INDEX.get(x))


def classify_finger_wise(references, incoming_frame):
    finger_joints = ((1, 2, 3, 4),
                     (6, 7, 8),
                     (10, 11, 12),
                     (14, 15, 16),
                     (18, 19, 20))
    if not incoming_frame: return [{'class': 'NA', 'distance': 'NA'}]

    selected_references = references
    selected_references['total_distance'] = 0
    for idx, finger in enumerate(finger_joints):
        col_names = [('{}_0'.format(joint), '{}_1'.format(joint), '{}_2'.format(joint)) for joint in finger]
        col_names = flatten_points(col_names)
        starting_index = (finger[0] - (idx + 1)) * 3
        end_index = starting_index + len(finger) * 3
        incoming_finger = incoming_frame[starting_index: end_index]
        selected_references['distance'] = selected_references.drop(['sign', 'distance'], errors='ignore', axis=1) \
            [col_names] \
            .apply(lambda x: distance.euclidean(x, incoming_finger), axis=1)
        selected_references['total_distance'] = selected_references['total_distance'] + selected_references['distance']
        selected_references = selected_references[selected_references.distance < 0.5]
        if selected_references.empty:
            logging.info('No matching sign at finger {}'.format(idx))
            return [{'class': 'NA', 'distance': 'NA'}]
    selected_references = selected_references.sort_values(by=['total_distance'])
    sign = LABEL_VS_INDEX.get(selected_references['sign'].iloc[0])
    logging.info('Sign is')
    return [{'class': sign}]

def rule_based_classify(pred, angles):
    pred_sign = pred.get('index')
    # 'U උ' and 'L ල්'
    if pred_sign == 7 or pred_sign == 27:
        z_rotation = angles[1]
        if z_rotation> 75: sign = 27
        else: sign = 7
        return  [{'class': LABEL_VS_INDEX.get(sign), 'index': sign}]
    # 'Dh ද්' and 'P ප්'
    elif pred_sign == 17 or pred_sign == 22:
        z_rotation = angles[1]
        if z_rotation> 75: sign = 17
        else: sign = 22
        return  [{'class': LABEL_VS_INDEX.get(sign), 'index': sign}]
    else: return [pred]