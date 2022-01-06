import logging
from itertools import chain

from sklearn.linear_model import LogisticRegression
from scipy.stats import mode

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import pygame
from scipy.spatial import distance
from sklearn import tree

from feature_extraction.pre_processor import un_flatten_points, get_angle_v2, normalize_flat_coordinates_scale
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
        return unified_cluster_means.reset_index(drop=True)

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
        mean_angles_df_cols = [str(point_pair) for point_pair in EDGE_PAIRS_FOR_ANGLES]
        mean_angles_df = pd.DataFrame(columns=mean_angles_df_cols)

        for row in means_list:
            angles_for_the_row = []
            row = un_flatten_points(row)
            for limb_pair in EDGE_PAIRS_FOR_ANGLES:
                limb2 = [row[limb_pair[1][1]][i] - row[limb_pair[1][0]][i] for i in range(0, 3)]
                limb1 = [row[limb_pair[0][1]][i] - row[limb_pair[0][0]][i] for i in range(0, 3)]
                reference_angle = get_angle_v2(limb2, limb1)
                angles_for_the_row.append((reference_angle-90) / 90)
            row_angles_df = pd.DataFrame([angles_for_the_row], columns=mean_angles_df_cols)
            mean_angles_df = mean_angles_df.append(row_angles_df)
        mean_angles_df = pd.concat((mean_angles_df.reset_index(drop=True), signs.rename('sign').reset_index(drop=True)), axis=1).drop('index', axis=1, errors='ignore')
        return mean_angles_df

    def unify_frame_features(self, landmark):
        angles = []
        for limb_pair in EDGE_PAIRS_FOR_ANGLES:
            limb2 = [landmark[limb_pair[1][1]][i] - landmark[limb_pair[1][0]][i] for i in range(0, 3)]
            limb1 = [landmark[limb_pair[0][1]][i] - landmark[limb_pair[0][0]][i] for i in range(0, 3)]
            angle = get_angle_v2(limb2, limb1)
            angles.append((angle-90) / 90)

        return angles


class ClassifierByAnglesAndCoordinates(NearestNeighborPoseClassifier):
    def __init__(self, cluster_means: pd.DataFrame, threshold, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        self.coordinateClassifier = ClassifierByFlatCoordinates(cluster_means, threshold, vertices_to_ignore)
        self.angleClassifier = ClassifierByAngles(cluster_means, threshold)
        super().__init__(cluster_means, 2)

    def unify_cluster_mean_features(self, cluster_means):
        unified_coordinates = self.coordinateClassifier.cluster_means.drop('sign', axis=1)
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
        self.clf0 = LogisticRegression(random_state=0).fit(self.cluster_means.drop('sign', axis=1), self.cluster_means['sign']) # - Good
        self.clf1 = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=4, weights='distance') # - Good
            #euc: 0.5689, 0.5027
            #manhattan: 0.5377, 0.4836
            #chebyshev: 0.5114, 0.4996
            #minkowski: 0.5689, 0.5027; p=3 0.5340, 0.4710; p=4 0.5737, 0.4908, p=4 weights=distance 0.5832, 0.5507
            #wminkowski:
            #seuclidean:
            #mahalanobis:
        self.clf2 = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini')
        self.clf3 = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')
        # - Good - 0.58 acc; random state not set
        # gini 0.5605, 0.54747 rs 0;
        # entropy 0.5473, 0.5303
        self.clf4 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
            max_depth=1, random_state=0)
        self.clf1.fit(self.cluster_means.drop('sign', axis=1), self.cluster_means['sign'])
        self.clf2.fit(self.cluster_means.drop('sign', axis=1), self.cluster_means['sign'])
        self.clf3.fit(self.cluster_means.drop('sign', axis=1), self.cluster_means['sign'])
        self.clf4.fit(self.cluster_means.drop('sign', axis=1), self.cluster_means['sign'])
        pass

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.unify_frame_features(landmark)
        # logging.info(incoming_frame)
        p0 = self.clf0.predict([incoming_frame, ])
        p1 = self.clf1.predict([incoming_frame, ])
        p2 = self.clf2.predict([incoming_frame, ])
        # p4 = self.clf4.predict([incoming_frame, ])
        p = mode([p0[0], p1[0], p2[0]])
        if p:
            p=p[0]
            prediction = [{'class': LABEL_VS_INDEX.get(p[0])}]
            return prediction
        else: return [{'class': 'NA', 'distance': 'NA'}]
        # if p4:
        #     prediction = [{'class': LABEL_VS_INDEX.get(p4[0])}]
        #     return prediction
        # else: return [{'class': 'NA', 'distance': 'NA'}]

