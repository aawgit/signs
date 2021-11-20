import csv
import logging

import numpy as np
import pandas as pd


def write_frame_vs_landmarks_to_csv(processed_q, output_file, no_of_features=21):
    logging.info('Writing to file {}'.format(output_file))
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", *[i for i in range(0, no_of_features)]])
        while True:
            try:
                if not processed_q.empty():
                    current_vertices, frame_no = processed_q.get()
                    writer.writerow([frame_no, *current_vertices])
                else:
                    pass
            except Exception as e:
                logging.error(e)
                break


def add_labels_to_landmarks(land_marks_vs_frame, label_vs_second_file, frame_rate=29.97):
    """

    :param land_marks_vs_frame_file:
    :param output_file:
    :return:
    """
    # TODO: Add dynamic or not

    label_vs_second_file.start = label_vs_second_file.apply(lambda row: int(row.start * frame_rate), axis=1)
    label_vs_second_file.end = label_vs_second_file.apply(lambda row: int(row.end * frame_rate), axis=1)

    label_vs_frame = label_vs_second_file
    # TODO: More efficient way of merging the two dfs
    sign_count = len(label_vs_frame.sign)

    def get_label(t):
        filtered_id_series = ((t >= label_vs_frame.start) & (t <= label_vs_frame.end))
        if filtered_id_series.any():
            label_idx = filtered_id_series.dot(np.arange(sign_count))
            return int(label_vs_frame.sign[label_idx])

    land_marks_vs_frame["sign"] = land_marks_vs_frame.frame.transform(get_label)
    land_marks_vs_frame = land_marks_vs_frame[land_marks_vs_frame['sign'].notna()]
    merged = pd.merge(land_marks_vs_frame, label_vs_frame, on='sign', how='left').drop(['start', 'end'], axis=1)
    return merged

if __name__ == '__main__':
    land_marks = pd.read_csv('../data/training/flatten_vertices_with_labels.csv')
    label_vs_second = pd.read_csv('../data/training/labels_Chaminda.csv')

    sign_vs_frame = add_labels_to_landmarks(land_marks, label_vs_second)
    output = '../data/training/sign_vs_landmark_t01.csv'
    sign_vs_frame.to_csv(output, index=False)