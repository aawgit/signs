import logging
import time as tm_mod
from itertools import chain

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.stats import mode
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as DTClassifier
from xgboost import XGBClassifier

from src.classification.classifier import plot_learning_curve, rule_based_classify
from src.feature_extraction.pre_processor import un_flatten_points, get_angle, flatten_points, pre_process
from src.utils.constants import EDGE_PAIRS_FOR_ANGLES, LABEL_VS_INDEX
from src.utils.helper import _get_landmark_col_names


class PoseClassifier:
    def __init__(self, training_data: list):
        lm_cols = _get_landmark_col_names()
        t_data_list = []
        for land_marks in training_data:
            lm_row = [land_marks[0]]
            for landmark_point in land_marks[1:]:
                lm_row.extend(np.round(landmark_point, 4))
            t_data_list.append(lm_row)
        t_data = pd.DataFrame(t_data_list, columns=['sign', *lm_cols])
        t_data[lm_cols] = t_data[lm_cols].apply(lambda row: pre_process_landmark(row)[0], axis=1)
        self.training_data = self.extract_training_data_features(t_data)
        self.X_train = self.training_data.drop('sign', axis=1).values
        self.y_train = self.training_data['sign'].values

    def extract_training_data_features(self, training_data):
        return pd.DataFrame()

    def extract_test_data_features(self, landmark):
        return []

    def classify(self, landmark):
        pass


class ClassifierByFlatCoordinates(PoseClassifier):
    def __init__(self, training_data: pd.DataFrame, vertices_to_ignore=None):
        if vertices_to_ignore is None:
            vertices_to_ignore = [0, 5, 9, 13, 17]
        self.vertices_to_ignore = vertices_to_ignore
        super().__init__(training_data)

    def extract_training_data_features(self, training_data: pd.DataFrame):
        # training_data = normalize_flat_coordinates_scale(training_data)
        if self.vertices_to_ignore:
            cols_to_drop = []
            for vertex in self.vertices_to_ignore:
                for i in range(0, 3):
                    cols_to_drop.append('{}_{}'.format(vertex, i))

            unified_training_data = training_data.drop(cols_to_drop, errors='ignore', axis=1)
        else:
            unified_training_data = training_data
        # unified_training_data = self._drop_z_axis(unified_training_data)
        return unified_training_data.reset_index(drop=True)

    def _drop_z_axis(self, training_data):
        cols_to_drop = [col for col in training_data.columns.values if col.endswith('2')]

        remaining_training_data = training_data.drop(cols_to_drop, errors='ignore', axis=1)
        return remaining_training_data

    def extract_test_data_features(self, landmark):
        landmark = list(landmark)
        landmark = pre_process_landmark(landmark)
        if self.vertices_to_ignore:
            for idx, vertex in enumerate(self.vertices_to_ignore):
                del landmark[vertex - idx]
        flattened_coordinates = list(chain(*landmark))
        return flattened_coordinates

    def remve_z_coordinate(self, flattened_coordinates):
        k = 3
        del flattened_coordinates[k - 1::k]
        return flattened_coordinates


class ClassifierByAngles(PoseClassifier):
    def __init__(self, training_data: pd.DataFrame, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        super().__init__(training_data)

    def get_angles(self, means_list, mean_angles_df_cols):
        mean_angles_df = pd.DataFrame(columns=mean_angles_df_cols)
        for row in means_list:
            angles_for_the_row = []
            row = un_flatten_points(row)
            for limb_pair in EDGE_PAIRS_FOR_ANGLES:
                limb2 = [row[limb_pair[1][1]][i] - row[limb_pair[1][0]][i] for i in range(0, 3)]
                limb1 = [row[limb_pair[0][1]][i] - row[limb_pair[0][0]][i] for i in range(0, 3)]
                reference_angle = get_angle(limb2, limb1)
                angles_for_the_row.append((reference_angle - 90) / 90)
            row_angles_df = pd.DataFrame([angles_for_the_row], columns=mean_angles_df_cols)
            mean_angles_df = mean_angles_df.append(row_angles_df)
        return mean_angles_df

    def _get_angles(self, landmarks):
        angles = []
        for limb_pair in EDGE_PAIRS_FOR_ANGLES:
            limb2 = [landmarks[limb_pair[1][1]][i] - landmarks[limb_pair[1][0]][i] for i in range(0, 3)]
            limb1 = [landmarks[limb_pair[0][1]][i] - landmarks[limb_pair[0][0]][i] for i in range(0, 3)]
            angle = get_angle(limb2, limb1)
            angles.append((angle - 90) / 90)

        return angles

    def extract_training_data_features(self, training_data):
        signs = training_data['sign']
        t_data_list = training_data.drop(['sign'], axis=1, errors='ignore').values.tolist()

        t_data_angles = []
        for row in t_data_list:
            angles = self._get_angles(row)
            t_data_angles.append(angles)

        t_data_df = pd.DataFrame(t_data_angles, columns= [str(point_pair) for point_pair in EDGE_PAIRS_FOR_ANGLES])

        mean_angles_df = pd.concat((t_data_df.reset_index(drop=True), signs.rename('sign').reset_index(drop=True)),
                                   axis=1).drop('index', axis=1, errors='ignore')
        return mean_angles_df

    def extract_test_data_features(self, landmarks):
        return self._get_angles(landmarks)


class ClassifierByAnglesAndCoordinates(PoseClassifier):
    def __init__(self, training_data: pd.DataFrame, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        self.coordinateClassifier = ClassifierByFlatCoordinates(training_data, vertices_to_ignore)
        self.angleClassifier = ClassifierByAngles(training_data)
        super().__init__(training_data)

    def extract_training_data_features(self, training_data):
        unified_coordinates = self.coordinateClassifier.training_data.drop(['sign'], axis=1, errors='ignore')
        unified_angles = self.angleClassifier.training_data
        unified_co_and_angle = pd.concat([unified_coordinates, unified_angles], axis=1)
        return unified_co_and_angle

    def extract_test_data_features(self, landmark):
        unified_coordinates = self.coordinateClassifier.extract_test_data_features(landmark)
        unified_angles = self.angleClassifier.extract_test_data_features(landmark)
        unified_coordinates.extend(unified_angles)
        return unified_coordinates


class DecisionTreeClassifier(ClassifierByAnglesAndCoordinates):
    """
    Not used
    """

    def __init__(self, training_data: pd.DataFrame, vertices_to_ignore=None):
        super().__init__(training_data)
        self.X = self.training_data.drop('sign', axis=1)
        self.Y = self.training_data['sign']
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(self.X, self.Y)

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.extract_test_data_features(landmark)
        self.clf.predict([incoming_frame])


class IndividualClassifier(ClassifierByAnglesAndCoordinates):
    def __init__(self, training_data: pd.DataFrame, model):
        super().__init__(training_data, None)
        self.model = model

    def tune_hpp(self):
        pass

    def plot_lc(self):
        fig, axes = plt.subplots(1, figsize=(10, 15))

        X, y = self.X_train, self.y_train

        title = "Learning Curve"
        # Cross validation with 50 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        # cv = ShuffleSplit(n_splits=4,  test_size=0.2, random_state=1)
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=30, random_state=1)
        estimator = self.model
        plot_learning_curve(
            estimator, title, X, y, ylim=(0.4, 1.01), cv=cv, n_jobs=4
        )

        plt.show()


class LogisticRegressionPoseClassifier(IndividualClassifier):
    def __init__(self, training_data: pd.DataFrame):
        # model = LogisticRegression(random_state=1, C=100, penalty='l2', solver='liblinear')
        # model = LogisticRegression(random_state=1, C=100, penalty='l2', solver='liblinear', max_iter=1000)  # hpp
        # model = LogisticRegression(multi_class='multinomial', random_state=1, C=100, penalty='l2', solver='newton-cg', max_iter=1000)# just before the exam
        model = LogisticRegression(multi_class='multinomial', random_state=1, C=100, penalty='l2', solver='newton-cg',
                                   max_iter=1000)  # after the exam
        super().__init__(training_data, model)

    def tune_hpp(self):
        def _tune_hpp(solvers: list, penalty: list, c_values: list, max_iter: list, multi_class: str):
            # example of grid searching key hyperparametres for logistic regression
            logging.info('Hyper-parameter search for LR classifier')
            # define dataset
            X, y = self.X_train, self.y_train  # make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
            # define models and parameters
            model = LogisticRegression()
            # define grid search
            grid = dict(multi_class=[multi_class], solver=solvers, penalty=penalty, C=c_values, max_iter=max_iter)
            cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=15, random_state=1)
            grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                       error_score=0)
            grid_result = grid_search.fit(X, y)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                logging.info("%f (%f) with: %r" % (mean, stdev, param))

        multiclass_options = ['ovr', 'multinomial']

        for idx, multi_class in enumerate(multiclass_options):
            # if idx == 0:
            #     solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            # if idx == 1:
            #     solvers = ['newton-cg', 'lbfgs', 'sag', 'saga']
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2', 'l1']
            c_values = [100, 50, 25, 10, 1.0, 0.1, 0.01]
            max_iter = [1000]

            _tune_hpp(solvers, penalty, c_values, max_iter, multi_class)


class RandomForestPoseClassifier(IndividualClassifier):
    def __init__(self, training_data: pd.DataFrame):
        model = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy', max_features='log2')
        super().__init__(training_data, model)

    def tune_hpp(self):
        # example of grid searching key hyperparameters for RandomForestClassifier
        logging.info('Hyper-parameter search for RF classifier')

        # define dataset
        X, y = self.X_train, self.y_train  # make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)
        # define models and parameters
        self.model = RandomForestClassifier()
        n_estimators = [10, 100]
        max_features = ['sqrt', 'log2']
        criterion = ['gini', 'entropy']
        # define grid search
        grid = dict(n_estimators=n_estimators, max_features=max_features, criterion=criterion)
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=15, random_state=1)
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(X, y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            logging.info("%f (%f) with: %r" % (mean, stdev, param))


class KNNPoseClassifier(IndividualClassifier):
    def __init__(self, training_data: pd.DataFrame):
        # model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', p=1, weights='distance') #hpp
        model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3, weights='distance')
        super().__init__(training_data, model)

    def tune_hpp(self):
        # example of grid searching key hyperparametres for KNeighborsClassifier
        logging.info('Hyper-parameter search for KNN classifier')
        # define dataset
        X, y = self.X_train, self.y_train
        # define models and parameters
        self.model = KNeighborsClassifier()
        n_neighbors = range(1, 4)
        weights = ['uniform', 'distance']
        metric = ['euclidean', 'manhattan', 'minkowski']
        p = [1, 2, 3, 4]
        # define grid search
        grid = dict(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p)
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=30, random_state=1)
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(X, y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            logging.info("%f (%f) with: %r" % (mean, stdev, param))


class XGBoostPoseClassifier(IndividualClassifier):
    def __init__(self, training_data: pd.DataFrame):
        # model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', p=1, weights='distance') #hpp
        model = XGBClassifier()
        super().__init__(training_data, model)

    def tune_hpp(self):
        # example of grid searching key hyperparametres for KNeighborsClassifier
        logging.info('Hyper-parameter search for XGBoost classifier')
        # define dataset
        X, y = self.X_train, self.y_train
        le = LabelEncoder()
        y = le.fit_transform(y)

        self.model = XGBClassifier()

        grid = {
            'n_estimators': [10, 100, 400],
            'max_depth': [2, 3],
            'learning_rate': [0.1, 0.20, 0.4],
            # 'min_child_weight': [1, 10, 100]
        }
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(X, y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            logging.info("%f (%f) with: %r" % (mean, stdev, param))


class DesicionTreePoseClassifier(IndividualClassifier):
    def __init__(self, training_data: pd.DataFrame):
        # model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', p=1, weights='distance') #hpp
        model = DTClassifier()
        super().__init__(training_data, model)

    def tune_hpp(self):
        # example of grid searching key hyperparametres for KNeighborsClassifier
        logging.info('Hyper-parameter search for Decision Tree classifier')
        # define dataset
        X, y = self.X_train, self.y_train
        # define models and parameters
        self.model = DTClassifier()
        criterion = ['gini', 'entropy', 'log_loss']
        max_features = ['auto', 'sqrt', 'log2']
        min_samples_split = [2, 3, 4]
        # metric = ['euclidean', 'manhattan', 'minkowski']
        # p = [1, 2, 3, 4]
        # define grid search
        grid = dict(criterion=criterion, max_features=max_features, min_samples_split=min_samples_split)
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=30, random_state=1)
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(X, y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            logging.info("%f (%f) with: %r" % (mean, stdev, param))


class NaiveBaysPoseClassifier(IndividualClassifier):
    def __init__(self, training_data: pd.DataFrame):
        # model = KNeighborsClassifier(n_neighbors=3, metric='euclidean', p=1, weights='distance') #hpp
        model = MultinomialNB()
        super().__init__(training_data, model)

    def tune_hpp(self):
        # example of grid searching key hyperparametres for KNeighborsClassifier
        logging.info('Hyper-parameter search for MultinomialNB classifier')
        # define dataset
        X, y = self.X_train, self.y_train
        # define models and parameters
        self.model = MultinomialNB()
        fit_prior = [True, False]

        grid = dict(fit_prior=fit_prior)
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=30, random_state=1)
        grid_search = GridSearchCV(estimator=self.model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',
                                   error_score=0)
        grid_result = grid_search.fit(X, y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            logging.info("%f (%f) with: %r" % (mean, stdev, param))


class CascadedClassifier(ClassifierByAnglesAndCoordinates):
    def __init__(self, training_data: list):
        super().__init__(training_data, None)
        # self.training_data = self.get_only_important_features_training(self.training_data)
        # X_train = self.training_data.drop('sign', axis=1).values
        # y_train = self.training_data['sign'].values
        # self.lr = LogisticRegression(random_state=0).fit(X_train, y_train)  # Documented
        self.lr = LogisticRegressionPoseClassifier(training_data).model  # hpp
        # self.lr = LogisticRegression(random_state=0,
        #                              C=10,
        #                              penalty='l2',
        #                              solver='liblinear').fit(X_train, y_train)  # - Good
        self.knn1 = KNNPoseClassifier(training_data).model  # - Documented
        # self.knn1 = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance')  # hpp
        # self.knn1 = KNeighborsClassifier(n_neighbors=1,
        #                                  metric='euclidean',
        #                                  weights='uniform')

        self.rf1 = RandomForestPoseClassifier(training_data).model  # Documented
        # self.rf1 = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini', max_features='sqrt') # hpp
        # self.rf1 = RandomForestClassifier(n_estimators=1000,
        #                                   random_state=0,
        #                                   criterion='entropy',
        #                                   max_features='sqrt')

        # self.rf2 = RandomForestClassifier(n_estimators=100, random_state=0, criterion='entropy')

        self.lr.fit(self.X_train, self.y_train)
        self.knn1.fit(self.X_train, self.y_train)
        self.rf1.fit(self.X_train, self.y_train)

        # self.clf4.fit(self.training_data.drop('sign', axis=1), self.training_data['sign'])
        pass

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.extract_test_data_features(landmark)
        # incoming_frame = self.get_only_important_features(incoming_frame)

        pred_knn = self.knn1.predict([incoming_frame, ])
        prob = self.knn1.predict_proba([incoming_frame, ])
        prob = max(*prob)
        if prob > 0.66:
            p = pred_knn[0]
        else:
            pred_lr = self.lr.predict([incoming_frame, ])
            pred_rf1 = self.rf1.predict([incoming_frame, ])
            # p4 = self.clf4.predict([incoming_frame, ])
            p = mode([pred_rf1[0], pred_lr[0], pred_knn[0]]).mode[0]

        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction

    def classify2(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.extract_test_data_features(landmark)
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

    def classify3(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.extract_test_data_features(landmark)

        pred_lr = self.lr.predict([incoming_frame, ])
        prob = self.lr.predict_proba([incoming_frame, ])
        prob = max(*prob)

        if prob > 0.6:
            p = pred_lr[0]
        else:
            pred_knn = self.knn1.predict([incoming_frame, ])
            pred_rf1 = self.rf1.predict([incoming_frame, ])
            p = mode([pred_rf1[0], pred_lr[0], pred_knn[0]]).mode[0]

        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction

    def classify4(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.extract_test_data_features(landmark)

        pred_lr = self.lr.predict([incoming_frame, ])
        prob = self.lr.predict_proba([incoming_frame, ])
        prob = max(*prob)

        if prob > 0.66:
            p = pred_lr[0]
        else:
            pred_knn = self.knn1.predict([incoming_frame, ])
            prob = self.knn1.predict_proba([incoming_frame, ])
            prob = max(*prob)
            if prob > 0.6:
                p = pred_knn[0]
            else:
                pred_rf1 = self.rf1.predict([incoming_frame, ])
                p = mode([pred_rf1[0], pred_lr[0], pred_knn[0]]).mode[0]

        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction

    def classify5(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.extract_test_data_features(landmark)

        pred_lr = self.lr.predict([incoming_frame, ])
        pred_knn = self.knn1.predict([incoming_frame, ])
        pred_rf1 = self.rf1.predict([incoming_frame, ])

        p = mode([pred_rf1[0], pred_lr[0], pred_knn[0]]).mode[0]

        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction

    def classify_cascaded(self, landmark, angles):
        pred = self.classify3(landmark)

        # This block is only used for ablation study using individual classifiers
        # if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        # incoming_frame = self.extract_test_data_features(landmark)
        # p = self.knn1.predict([incoming_frame, ])
        # pred = [{'class': LABEL_VS_INDEX.get(p[0]), 'index': p}]

        if pred[0].get('class') == 'NA':
            return
        pred = rule_based_classify(pred[0], angles)
        return pred

    def hpp_level1_ensemble(self, training_data):

        X, y = self.X_train, self.y_train
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=15, random_state=1)
        all_results = []
        for train_index, test_index in cv.split(X, y):
            self.knn1.fit(X[train_index], y[train_index])
            self.rf1.fit(X[train_index], y[train_index])
            self.lr.fit(X[train_index], y[train_index])
            test_data = training_data.drop('sign', axis=1)
            results = []
            for row_index in test_index:
                row = test_data.iloc[row_index]
                landmark = un_flatten_points(row)
                predicted = self.classify3(landmark)
                results.append(dict(true=training_data.iloc[row_index].sign, predicted=predicted[0].get('index')))
            all_results.extend(results)
        all_results_df = pd.DataFrame(all_results)
        acc = accuracy_score(all_results_df['true'], all_results_df['predicted'])
        logging.info(acc)

    def hpp_cascaded(self, training_data):

        X, y = self.X_train, self.y_train
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=15, random_state=1)
        all_results = []
        for train_index, test_index in cv.split(X, y):
            self.knn1.fit(X[train_index], y[train_index])
            self.rf1.fit(X[train_index], y[train_index])
            self.lr.fit(X[train_index], y[train_index])
            test_data = training_data.drop('sign', axis=1)
            results = []
            for row_index in test_index:
                row = test_data.iloc[row_index]
                landmark = un_flatten_points(row)
                predicted = self.classify_cascaded(landmark)
                results.append(dict(true=training_data.iloc[row_index].sign, predicted=predicted[0].get('index')))
            all_results.extend(results)
        all_results_df = pd.DataFrame(all_results)
        acc = accuracy_score(all_results_df['true'], all_results_df['predicted'])
        logging.info(acc)


class EnsembleClassifierTwo(PoseClassifier):
    """
    Not selected for the thesis
    """

    def __init__(self, training_data: pd.DataFrame, threshold, vertices_to_ignore=None):
        self.vertices_to_ignore = vertices_to_ignore
        self.coordinateClassifier = ClassifierByFlatCoordinates(training_data, vertices_to_ignore)
        self.angleClassifier = ClassifierByAngles(training_data)

        super().__init__(training_data)

        X_train_c = self.coordinateClassifier.training_data.drop(['sign', 'source'], axis=1, errors='ignore').values
        y_train_c = self.coordinateClassifier.training_data['sign'].values

        self.lr_c = LogisticRegression(random_state=0).fit(X_train_c, y_train_c)  # - Good
        self.knn1_c = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=4, weights='distance')  # - Good
        self.rf1_c = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini')

        self.lr_c.fit(X_train_c, y_train_c)
        self.knn1_c.fit(X_train_c, y_train_c)
        self.rf1_c.fit(X_train_c, y_train_c)

        X_train_a = self.angleClassifier.training_data.drop('sign', axis=1).values
        y_train_a = self.angleClassifier.training_data['sign'].values

        self.lr_a = LogisticRegression(random_state=0).fit(X_train_a, y_train_a)  # - Good
        self.knn1_a = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=4, weights='distance')  # - Good
        self.rf1_a = RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini')

        self.lr_a.fit(X_train_a, y_train_a)
        self.knn1_a.fit(X_train_a, y_train_a)
        self.rf1_a.fit(X_train_a, y_train_a)

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame_c = self.coordinateClassifier.extract_test_data_features(landmark)

        pred_knn_c = self.knn1_c.predict([incoming_frame_c, ])
        pred_lr_c = self.lr_c.predict([incoming_frame_c, ])
        pred_rf1_c = self.rf1_c.predict([incoming_frame_c, ])

        incoming_frame_a = self.angleClassifier.extract_test_data_features(landmark)
        pred_knn_a = self.knn1_a.predict([incoming_frame_a, ])
        pred_lr_a = self.lr_a.predict([incoming_frame_a, ])
        pred_rf1_a = self.rf1_a.predict([incoming_frame_a, ])

        m1 = mode([pred_rf1_a[0],
                   pred_lr_a[0],
                   pred_knn_a[0]])
        if m1[1][0] > 1:
            p = m1[0][0]
        else:
            m2 = mode([pred_rf1_c[0],
                       pred_lr_c[0],
                       pred_knn_c[0]])

            p = m2[0][0]

        prediction = [{'class': LABEL_VS_INDEX.get(p), 'index': p}]
        return prediction


class FingerwiseCompareClassifier(ClassifierByAnglesAndCoordinates):
    """
    Not selected for the thesis
    """

    def __init__(self, training_data: pd.DataFrame):
        super().__init__(training_data)
        self.finger_joints = ((1, 2, 3, 4),
                              (6, 7, 8),
                              (10, 11, 12),
                              (14, 15, 16),
                              (18, 19, 20))

    def classify(self, landmark):
        if not landmark: return [{'class': 'NA', 'distance': 'NA'}]
        incoming_frame = self.extract_test_data_features(landmark)
        prediction = self.classify_finger_wise(incoming_frame, self.training_data)
        return prediction

    def classify_finger_wise(self, references, incoming_frame):
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
            selected_references['total_distance'] = selected_references['total_distance'] + selected_references[
                'distance']
            selected_references = selected_references[selected_references.distance < 0.5]
            if selected_references.empty:
                logging.info('No matching sign at finger {}'.format(idx))
                return [{'class': 'NA', 'distance': 'NA'}]
        selected_references = selected_references.sort_values(by=['total_distance'])
        sign = LABEL_VS_INDEX.get(selected_references['sign'].iloc[0])
        logging.info('Sign is')
        return [{'class': sign}]


def tune_hpp_classifier_option(dataset: pd.DataFrame):
    def classify(model, i, row):
        label = row[-1]
        row = row[:-1]
        row = un_flatten_points(list(row))
        start = tm_mod.process_time()
        if i == 1:
            prediction = model.classify(row)
        elif i == 2:
            prediction = model.classify2(row)
        elif i == 3:
            prediction = model.classify3(row)
        elif i == 4:
            prediction = model.classify4(row)
        else:
            prediction = model.classify5(row)
        end = tm_mod.process_time()
        duration = end - start
        if prediction[0].get('class') == 'NA':
            return
        if label in [27, 17, 51]:
            angles = [0, 80]
        else:
            angles = [0, 0]
        # prediction = rule_based_classify(prediction[0], angles)
        prediction[0].update({'truth_sign': LABEL_VS_INDEX.get(label), 'time': duration})
        return prediction[0]

    dataset.reset_index(drop=True, inplace=True)
    X, y = dataset.drop('sign', axis=1).to_numpy(), dataset['sign']
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=15, random_state=1)
    columns = []
    for vertex in range(0, 21):
        for i in range(0, 3):
            columns.append('{}_{}'.format(vertex, i))

    measurements = {1: {'acc': [], 'pre': [], 'time': []},
                    2: {'acc': [], 'pre': [], 'time': []},
                    3: {'acc': [], 'pre': [], 'time': []},
                    4: {'acc': [], 'pre': [], 'time': []},
                    5: {'acc': [], 'pre': [], 'time': []}}
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        training_set = pd.concat([pd.DataFrame(X_train, columns=columns),
                                  (pd.DataFrame(y_train, columns=['sign']).reset_index(drop=True))], axis=1)
        test_set = pd.concat([pd.DataFrame(X_test, columns=columns),
                              (pd.DataFrame(y_test, columns=['sign']).reset_index(drop=True))], axis=1)

        # training_set = training_set[
        #     (training_set.sign != 7) & (training_set.sign != 17) & (training_set.sign != 30)]

        for i in [3, 5]:  # range(1, 6):
            model = CascadedClassifier(training_set)
            test_set.drop('prediction', axis=1, inplace=True, errors='ignore')
            test_set['prediction'] = test_set.apply(lambda x: classify(model, i, x), axis=1)
            results = pd.DataFrame(list(test_set.prediction))
            acc = accuracy_score(results['truth_sign'], results['class'])
            pre = precision_score(results['truth_sign'], results['class'], average='macro')
            logging.info('acc: {}\npre: {}'.format(acc, pre))
            measurements.get(i).get('acc').append(acc)
            measurements.get(i).get('pre').append(pre)
            time_mean = np.mean(results.time)
            measurements.get(i).get('time').append(time_mean)

            # plot_cnf_matrix(results)
    for i in [3, 5]:  # range(1, 6):
        logging.info('{} {} {} {}'.format(i,
                                          np.mean(measurements.get(i).get('acc')),
                                          np.mean(measurements.get(i).get('pre')),
                                          np.mean(measurements.get(i).get('time'))
                                          ))


def pre_process_landmark(un_flattened):
    """
    Takes a landmkark
    """
    pre_processed, angles = pre_process(un_flattened)
    flattened = flatten_points(pre_processed)
    return pd.Series(flattened), angles