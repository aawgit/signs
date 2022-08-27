import pandas as pd

from src.feature_extraction.pre_processor import flatten_points
from src.utils.constants import EDGE_PAIRS_FOR_ANGLES
from test.dummies import pre_processing_out
from src.archived.classifier import ClassifierByAngles


def test_get_angles():
    mean_angles_df_cols = [str(point_pair) for point_pair in EDGE_PAIRS_FOR_ANGLES]

    clf = ClassifierByAngles(pd.DataFrame(columns=['sign']))
    input = flatten_points(pre_processing_out)
    mean_angles_df = clf.get_angles([input], mean_angles_df_cols)

    angles_correct = [{'((5, 0), (0, 2))': -0.6630432206723899, '((0, 1), (1, 2))': 0.772871950493732,
                       '((1, 2), (2, 3))': 0.9233680073492201, '((2, 3), (3, 4))': 0.9522319327595259,
                       '((0, 5), (5, 6))': 0.8717345775704984, '((5, 6), (6, 7))': 0.8710415574756047,
                       '((6, 7), (7, 8))': 0.963498819556722, '((0, 9), (9, 10))': 0.9269417355337715,
                       '((9, 10), (10, 11))': 0.8890386105269651, '((10, 11), (11, 12))': 0.9885493572167727,
                       '((0, 13), (13, 14))': 0.8876935492493854, '((13, 14), (14, 15))': 0.9154606266838414,
                       '((14, 15), (15, 16))': 0.9617998837287669, '((0, 17), (17, 18))': 0.7878340284391707,
                       '((17, 18), (18, 19))': 0.9429335921915588, '((18, 19), (19, 20))': 0.9382450463655403}]
    result = mean_angles_df.to_dict('records')
    assert result[0].get('((0, 5), (0, 2))')==angles_correct[0].get('((0, 5), (0, 2))')
