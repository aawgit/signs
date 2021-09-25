import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def create_labels():
    # TODO: Add dynamic or not
    labels = pd.read_csv('./data/training/labels_Chaminda .csv')
    angles = pd.read_csv('./data/training/angles_Chaminda.csv')

    labels.start = labels.apply(lambda row: int(row.start * 29.97), axis=1)
    labels.end = labels.apply(lambda row: int(row.end * 29.97), axis=1)

    # TODO: More efficient way of merging the two dfs
    sign_count = len(labels.sign)

    def get_label(t):
        filtered_id_series = ((t >= labels.start) & (t <= labels.end))
        if filtered_id_series.any():
            label_idx = filtered_id_series.dot(np.arange(sign_count))
            return int(labels.sign[label_idx])

    angles["sign"] = angles.frame.transform(get_label)
    angles = angles[angles['sign'].notna()]
    merged = pd.merge(angles, labels, on='sign', how='left')[['frame', 'thumb_angle' ,'index_angle' , 'middle_angle',
                                                             'ring_angle' , 'pinky_angle' ,'sign', 'sign_character', 'dynamic']]
    merged.to_csv('./data/training/angles_labels_chaminda.csv', index=False)

def add_dynamic_or_not():
    labels = pd.read_csv('./data/training/labels_Chaminda .csv')
    angles_and_labels = pd.read_csv('./data/training/angles_labels_chaminda_2.csv')
    merged = pd.merge(angles_and_labels, labels[['sign', 'dynamic']], left_on='label', right_on='sign', how='left')
    merged.dropna(subset=['label'], how='all', inplace=True)
    merged.to_csv('./data/training/angles_labels_chaminda_3.csv', index=False)

def edge_cleaned_mean(x):
    length = len(x)
    starting = int(length*.1)
    ending = int(length*.9)
    x = x.iloc[starting:ending].mean()
    return x

def edge_cleaned_std(x):
    length = len(x)
    starting = int(length*.2)
    ending = int(length*.8)
    x = x.iloc[starting:ending].std()
    return x

if __name__ == '__main__':
    # add_dynamic_or_not()
    # angles_and_labels = pd.read_csv('./data/training/angles_labels_chaminda_3.csv')
    # angles_and_labels = angles_and_labels[angles_and_labels.dynamic!=1]
    # angles_and_labels.groupby('sign').agg(edge_cleaned_std)

    means = pd.read_csv('./data/training/means.csv')[['thumb_angle' ,'index_angle' , 'middle_angle',
                                                             'ring_angle' , 'pinky_angle']]
    cosine_similarity(means)




