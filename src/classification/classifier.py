import logging

import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt, font_manager as fm

from sklearn.linear_model import LogisticRegression
from scipy.stats import mode

from sklearn.model_selection import learning_curve, RepeatedStratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np

# from src.classify_entry import change_matplotlib_font
from src.utils.constants import LABEL_VS_INDEX


def _select_data_by_index(ar, indices):
    selected = [ar[idx] for idx in indices]
    return selected


class LevelOneClassifier:
    def __init__(self, X_train: list, y_train: list):
        self.X_train = X_train
        self.y_train = y_train
        self.lr = LogisticRegression(
            multi_class="multinomial",
            random_state=1,
            C=100,
            penalty="l2",
            solver="newton-cg",
            max_iter=1000,
        )
        self.knn = KNeighborsClassifier(
            n_neighbors=3, metric="minkowski", p=3, weights="distance"
        )
        self.rf = RandomForestClassifier(
            n_estimators=100, random_state=0, criterion="entropy", max_features="log2"
        )

    def train(self):
        self.lr.fit(self.X_train, self.y_train)
        self.knn.fit(self.X_train, self.y_train)
        self.rf.fit(self.X_train, self.y_train)

    def hpt(self):
        X, y = self.X_train, self.y_train
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=15, random_state=1)
        all_results = []
        for train_index, test_index in cv.split(X, y):
            self.knn.fit(
                _select_data_by_index(X, train_index),
                _select_data_by_index(y, train_index),
            )
            self.rf.fit(
                _select_data_by_index(X, train_index),
                _select_data_by_index(y, train_index),
            )
            self.lr.fit(
                _select_data_by_index(X, train_index),
                _select_data_by_index(y, train_index),
            )
            results = []
            for row_index in test_index:
                landmarks = X[row_index]
                predicted = self.classify(landmarks)
                results.append(
                    dict(true=y[row_index], predicted=predicted[0].get("index"))
                )
            all_results.extend(results)
        all_results_df = pd.DataFrame(all_results)
        acc = accuracy_score(all_results_df["true"], all_results_df["predicted"])
        logging.info(acc)

    def classify(self, feature_vector):
        if not feature_vector:
            return [{"class": "NA", "distance": "NA"}]

        pred_lr = self.lr.predict(
            [
                feature_vector,
            ]
        )
        prob = self.lr.predict_proba(
            [
                feature_vector,
            ]
        )
        prob = max(*prob)

        if prob > 0.6:
            p = pred_lr[0]
        else:
            pred_knn = self.knn.predict(
                [
                    feature_vector,
                ]
            )
            pred_rf1 = self.rf.predict(
                [
                    feature_vector,
                ]
            )
            p = mode([pred_rf1[0], pred_lr[0], pred_knn[0]]).mode[0]

        prediction = [{"class": LABEL_VS_INDEX.get(p), "index": p}]
        return prediction


class MultiLevelClassifier:
    def __init__(self, X_train: list, y_train: list):
        logging.info("Initializing classifier...")
        self.level_one_classifier = LevelOneClassifier(X_train, y_train)
        self.level_one_classifier.train()
        logging.info("Classifier trained.")

    def classify(self, landmarks, orientation):
        level_one_pred = self.level_one_classifier.classify(landmarks)
        if level_one_pred[0].get("class") == "NA":
            return
        pred = rule_based_classify(level_one_pred[0], orientation)
        return pred


def rule_based_classify(pred, angles):
    pred_sign = pred.get("index")
    # 'U උ' and 'L ල්'
    if pred_sign == 7 or pred_sign == 27:
        z_rotation = angles[1]
        if z_rotation > 45:
            sign = 27
        else:
            sign = 7
        return [{"class": LABEL_VS_INDEX.get(sign), "index": sign}]
    # 'Dh ද්' and 'P ප්'
    elif pred_sign == 17 or pred_sign == 22:
        z_rotation = angles[1]
        if z_rotation > 45:
            sign = 17
        else:
            sign = 22
        return [{"class": LABEL_VS_INDEX.get(sign), "index": sign}]
        # 'H හ්' and 'AW ඖ'
    elif pred_sign == 30 or pred_sign == 51:
        z_rotation = angles[1]
        if z_rotation > 45:
            sign = 51
        else:
            sign = 30
        return [{"class": LABEL_VS_INDEX.get(sign), "index": sign}]
    else:
        return [pred]


def plot_cnf_matrix(all_results):
    all_results.truth_sign = all_results["truth_sign"].apply(lambda x: x.split(" ")[1])
    all_results["class"] = all_results["class"].apply(lambda x: x.split(" ")[1])
    change_matplotlib_font("font_download_url")

    cf_matrix = confusion_matrix(
        all_results["truth_sign"],
        all_results["class"],
        labels=all_results["truth_sign"].unique(),
    )

    cf_matrix = cf_matrix.astype("float") / cf_matrix.sum(axis=1)[:, np.newaxis]
    ## Display the visualization of the Confusion Matrix.
    df_cm = pd.DataFrame(
        cf_matrix,
        index=all_results["truth_sign"].unique(),
        columns=all_results["truth_sign"].unique(),
    )
    ax = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt=".1f")
    # ax.set_title('Confusion Matrix for Test Data\n\n')
    ax.set_xlabel("\nPredicted Category")
    ax.set_ylabel("Actual Category ")
    # plt.rc('axes', unicode_minus=False)
    # plt.rc('font', **{'sans-serif' : 'Arial',
    #                      'family' : 'sans-serif'})
    plt.show()


def change_matplotlib_font(font_download_url):
    FONT_PATH = "utils/fonts/Yaldevi"

    # font_download_cmd = f"wget {font_download_url} -O {FONT_PATH}.zip"
    # unzip_cmd = f"unzip -o {FONT_PATH}.zip -d {FONT_PATH}"
    # # os.system(font_download_cmd)
    # os.system(unzip_cmd)

    font_files = fm.findSystemFonts(fontpaths=FONT_PATH)
    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    font_name = fm.FontProperties(fname=font_files[0]).get_name()
    matplotlib.rc("font", family=font_name)
    print("font family: ", plt.rcParams["font.family"])


def dtw_test():
    pass


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, "o-")
    # axes[1].fill_between(
    #     train_sizes,
    #     fit_times_mean - fit_times_std,
    #     fit_times_mean + fit_times_std,
    #     alpha=0.1,
    # )
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")
    #
    # # Plot fit_time vs score
    # fit_time_argsort = fit_times_mean.argsort()
    # fit_time_sorted = fit_times_mean[fit_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    # test_scores_std_sorted = test_scores_std[fit_time_argsort]
    # axes[2].grid()
    # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    # axes[2].fill_between(
    #     fit_time_sorted,
    #     test_scores_mean_sorted - test_scores_std_sorted,
    #     test_scores_mean_sorted + test_scores_std_sorted,
    #     alpha=0.1,
    # )
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt
