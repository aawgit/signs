import glob
import logging
import ast

import pandas as pd
import numpy as np
from scipy.spatial import distance
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from src.classification.classifier import change_matplotlib_font
from src.classification.classify_entry import get_training_data
from src.feature_extraction.pre_processor import pre_process

from src.pose_estimation.media_pipe_static_estimator import static_images
from src.utils.constants import LABEL_VS_INDEX


def add_dynamic_or_not():
    labels = pd.read_csv('./data/training/labels_Chaminda .csv')
    angles_and_labels = pd.read_csv('../../data/training/angles_labels_chaminda_2.csv')
    merged = pd.merge(angles_and_labels, labels[['sign', 'dynamic']], left_on='label', right_on='sign', how='left')
    merged.dropna(subset=['label'], how='all', inplace=True)
    merged.to_csv('./data/training/angles_labels_chaminda_3.csv', index=False)


def edge_cleaned_mean(x):
    length = len(x)
    starting = int(length * .2)
    ending = int(length * .8)
    x = x.iloc[starting:ending].mean()
    return x


def edge_cleaned_std(x):
    length = len(x)
    starting = int(length * .2)
    ending = int(length * .8)
    x = x.iloc[starting:ending].std()
    return x


def _get_file_name(image_index):
    index_str = str(image_index)
    index_len = len(index_str)
    gap = 6 - index_len
    prefix = ''
    for i in range(gap, 6):
        prefix = prefix + '0'
    return prefix + index_str + '.jpg'


def _get_land_marks(subject, scene, sign, root_folder):
    path = '{}/labels-final-revised1/subject{}/Scene{}/*.csv'.format(root_folder, subject, scene)
    files = glob.glob(path)
    df = pd.DataFrame()
    for f in files:
        csv = pd.read_csv(f, header=None)
        csv['group'] = f[-5]
        df = df.append(csv)
    if df.empty: return [], []
    range_for_sign = df[df[0] == sign]
    if range_for_sign.empty: return [], []
    start_image_index, end_image_index = range_for_sign[1].iloc[0].item(), range_for_sign[2].iloc[0].item()
    group = range_for_sign['group'].iloc[0]
    image_folder = '{}/images_320-240_1/Subject{}/Scene{}/Color/rgb{}/'.format(root_folder, subject, scene, group)

    land_marks_list = []
    indices_list = []
    for image_index in range(int(start_image_index), int(end_image_index)):
        file_name = _get_file_name(image_index)
        image_path = image_folder + file_name
        try:
            land_marks = static_images(image_path)
            # img = Image.open(image_path)
            # img.show()
            if land_marks:
                land_marks_list.append(land_marks)
                indices_list.append(image_index)
        except Exception as e:
            logging.error(e)
    return land_marks_list, indices_list


def get_land_marks(signers, signs, scenes):
    all_land_marks = pd.DataFrame()
    for sign in signs:
        sings_from_all_signers = []
        all_indices = []
        for signer in signers:
            for scene in scenes:
                land_marks, image_indices = _get_land_marks('0{}'.format(signer), scene, sign,
                                                            '/home/aka/Downloads/ego hands dataset')
                sings_from_all_signers.extend(land_marks)
                all_indices.extend(image_indices)

        processed_land_marks = []
        for land_marks in sings_from_all_signers:
            processed = pre_process(land_marks)
            processed_land_marks.append(processed)

        df = pd.DataFrame(processed_land_marks)
        df['sign'] = sign
        df['image_index'] = all_indices
        all_land_marks = all_land_marks.append(df)
    return all_land_marks


def group_by_sign_and_mean(land_marks: pd.DataFrame):
    # signs = range(24, 35)
    def _means(x):
        sum = 0, 0, 0
        for row in x:
            sum = sum[0] + row[0], sum[1] + row[1], sum[2] + row[2]
        total = len(x)
        return sum[0] / total, sum[1] / total, sum[2] / total

    means = land_marks.loc[:, land_marks.columns != 'image_index'].groupby('sign').agg(_means)
    return means


def get_mean_land_mark(land_marks: pd.DataFrame):
    # signs = range(24, 35)
    def _means(x):
        sum = 0, 0, 0
        for row in x:
            sum = sum[0] + row[0], sum[1] + row[1], sum[2] + row[2]
        total = len(x)
        return sum[0] / total, sum[1] / total, sum[2] / total

    means = land_marks.loc[:, land_marks.columns != 'image_index'].groupby('sign').agg(_means)
    return means.values.flatten().tolist()


def distance_from_mean(mean_land_mark: list, land_marks: pd.DataFrame):
    def _distance_from_mean(x):
        distance_row = []
        i = 0
        for column in x:
            if i == 21: break
            distance_from_mean = distance.euclidean(column, mean_land_mark[i])
            i = i + 1
            distance_row.append(distance_from_mean)
        return distance_row

    distances = land_marks.apply(lambda x: _distance_from_mean(x), axis=1)
    distances_df = pd.DataFrame(list(distances))
    # distances_df['image_index'] = land_marks['image_index']
    return distances_df


def get_variance(mean_land_mark, land_marks):
    distance = distance_from_mean(mean_land_mark, land_marks)
    squares = distance.apply(lambda x: np.square(x))
    variances = squares.sum(axis=0) / distance.shape[0]
    return variances

converter={"0": ast.literal_eval,
                                         "1": ast.literal_eval,
                                         "2": ast.literal_eval,
                                         "3": ast.literal_eval,
                                         "4": ast.literal_eval,
                                         "5": ast.literal_eval,
                                         "6": ast.literal_eval,
                                         "7": ast.literal_eval,
                                         "8": ast.literal_eval,
                                         "9": ast.literal_eval,
                                         "10": ast.literal_eval,
                                         "11": ast.literal_eval,
                                         "12": ast.literal_eval,
                                         "13": ast.literal_eval,
                                         "14": ast.literal_eval,
                                         "15": ast.literal_eval,
                                         "16": ast.literal_eval,
                                         "17": ast.literal_eval,
                                         "18": ast.literal_eval,
                                         "19": ast.literal_eval,
                                         "20": ast.literal_eval,
                                         }

def cluster_kmeans():
    land_marks = pd.read_csv('data/training/ego_hands_land_marks.csv',
                             converters=converter)
    # mean_land_marks = group_by_sign_and_mean(land_marks)

    flat_features = split_points_to_coordinates(land_marks)
    X_for_k_means = flat_features.drop(['sign', '0_0', '0_1', '0_2'], axis=1)
    kmeans = KMeans(n_clusters=12, random_state=0).fit(X_for_k_means)
    # distance_from_mean(list(mean_land_marks.loc[24]), land_marks[land_marks.sign == 24])
    # get_variance(list(mean_land_marks.loc[24]), land_marks[land_marks.sign == 24])
    # distances = distance_from_mean(list(mean_land_marks.loc[25]), land_marks[land_marks.sign == 25])

    # mean_land_mark = get_mean_land_mark(land_marks)
    X_dist = kmeans.transform(X_for_k_means)
    X_dist = pd.DataFrame(X_dist)

    def nan_all_but_min(df):
        arr = df.values
        idx = np.argmin(arr, axis=1)
        newarr = np.full_like(arr, np.nan, dtype='float')
        newarr[np.arange(arr.shape[0]), idx] = arr[np.arange(arr.shape[0]), idx]
        df = pd.DataFrame(newarr, columns=df.columns, index=df.index)
        return df

    X_dist = nan_all_but_min(X_dist)

    df = X_dist.melt()  # .
    plt.scatter(df.variable, df.value)

    # plt.hist(x, density=True, bins=30)

    # colors = np.where(X_dist["Animation"] == 1, 'y', 'k')
    # X_dist.plot.scatter(x="year", y="length", c=colors)
    plt.show()

def split_points_to_coordinates(land_marks: pd.DataFrame):
    new_df = pd.DataFrame()
    for column in land_marks:
        if column == 'sign' or column == 'image_index': continue
        new_col_list = ['{}_{}'.format(column, n) for n in range(0, 3)]
        for n, col in enumerate(new_col_list):
            new_df[col] = land_marks[column].apply(lambda point: point[n])
    new_df['sign'] = land_marks['sign']
    return new_df

def plot_training_data_histogram():
    training_data = get_training_data(with_origins=True)
    x = [LABEL_VS_INDEX.get(s_index) for s_index in list(training_data['sign'])]
    # plt.hist(x, bins=len(x))
    # plt.show()
    training_data['sign_2'] = x

    # training_data['sign_2'].value_counts().plot(kind='bar')

    ## Uncomment these for plot1
    # signs_unique = list(training_data['sign_2'].unique())
    # categories = training_data['sign_2'].value_counts()[signs_unique].index
    # counts = training_data['sign_2'].value_counts()[signs_unique].values
    # change_matplotlib_font('font_download_url')
    # plt.bar(categories, counts, width=0.5)
    # plt.title('Distribution of Training Data')
    # plt.xlabel('Sign')
    # plt.ylabel('Count')
    # plt.show()

    goaldf = pd.concat([training_data, pd.get_dummies(training_data.subject)], axis=1)[['sign_2','subject01', 'subject02', 'subject03', 'subject04']].groupby('sign_2').sum().reset_index()
    b = []
    colors = plt.cm.get_cmap('jet',5)
    xticks = [i for i in range(len(goaldf))]

    columns = goaldf.columns.values

    fig, ax = plt.subplots(1, 1)
    for i in range(1, len(columns)):
        if i == 1:
            bar_bottom = 0
        else:
            bar_bottom = bar_bottom + goaldf[columns[i - 1]].values
        b.append(plt.bar(xticks,
                         goaldf[columns[i]].values,
                         bottom=bar_bottom,
                         color=colors(i)))
        for i in range(len(b)):
            # ax.bar_label(b[i],
            #              padding=0,
            #              label_type='center',
            #              rotation='horizontal')
            ax.set_ylabel('Goal Contributions')
    ax.set_xlabel('Players')
    ax.set_xticks(xticks)
    ax.set_xticklabels(goaldf['sign_2'].values, rotation=45)  # , rotation_mode = 'anchor')
    ax.set_title('Top Ten Goal Contributions in 2020-2021')
    ax.legend(b, columns[1:])
    plt.show()

if __name__ == '__main__':
    land_mark_vs_label = pd.read_csv('./data/training/sign_vs_landmark_t01.csv', converters=converter).drop(['frame', 'sign_character'], axis=1, errors='ignore')
    land_mark_vs_label = land_mark_vs_label[land_mark_vs_label.dynamic!=1].drop(['dynamic'], axis=1, errors='ignore')
    land_mark_vs_label = split_points_to_coordinates(land_mark_vs_label)
    means = land_mark_vs_label.groupby('sign').agg(edge_cleaned_mean)
    means.to_csv('./data/training/means_cham_vertices_28_10_21_1.csv')

    # add_dynamic_or_not()
    # angles_and_labels = pd.read_csv('./data/training/angles_labels_chaminda_3.csv')
    # angles_and_labels = angles_and_labels[angles_and_labels.dynamic!=1]
    # angles_and_labels.groupby('sign').agg(edge_cleaned_std)

    # means = pd.read_csv('./data/training/means.csv')[['thumb_angle' ,'index_angle' , 'middle_angle',
    #                                                          'ring_angle' , 'pinky_angle']]
    # cosine_similarity(means)

    # Ego hands
    # land_marks = get_land_marks(list(range(1, 10)), list(range(24, 36)), list(range(1, 7)))

    # distances2 = distance_from_mean(mean_land_mark, land_marks)
    x = 5
    # for sign in signs:
    #     get_coordinates('01', '1', sign, '/home/aka/Downloads/ego hands dataset')
