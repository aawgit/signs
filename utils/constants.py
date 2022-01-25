EDGES_CUBE = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7)
)

VERTICES_CUBE = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)

LABEL_VS_INDEX = {
    1: 'A අ',
    2: 'AA ආ',
    3: 'AE ඇ',
    4: 'AEE ඈ',
    5: 'I ඉ',
    6: 'IEE ඊ',
    7: 'U උ',
    8: 'UU ඌ',
    9: 'E එ',
    10: 'EE ඒ',
    11: 'O ඔ',
    12: 'OE ඕ',
    13: 'K ක්',
    14: 'G ග්',
    15: 'J ජ්',
    16: 'T ට්',
    17: 'Dh ද්',
    18: 'NH ණ්',
    19: 'Th ත්',
    20: 'D ඩ්',
    21: 'N න්',
    22: 'P ප්',
    23: 'B බ්',
    24: 'M ම්',
    25: 'Y ය්',
    26: 'R ර්',
    27: 'L ල්',
    28: 'W ව්',
    29: 'S ස්',
    30: 'H හ්',
    31: 'LH ළ්',
    32: 'N(o) o',
    33: 'NG ඟ',
    34: 'ND ඬ්',
    35: 'NDh ඳ',
    36: 'MB ඹ්',
    37: 'KH ඛ්',
    38: 'GH ඝ්',
    39: 'DH ඪ්',
    40: 'PH ඵ්',
    41: 'DhH ධ්',
    42: 'TH ඨ්',
    43: 'Ch ච්',
    44: 'ChH ඡ්',
    45: 'YA <yansaya>',
    46: 'F ෆ්',
    47: 'BH භ්',
    48: 'ThH ථ්',
    49: 'AY ඓ',
    50: 'RU රු',
    51: 'AW ඖ',
    52: 'EI',
    53: 'SH',
    54: 'PSH',
    55: 'KDH',
    56: 'KDh',
    57: 'NGDh',
    58: 'RR', }

EDGE_PAIRS_FOR_ANGLES = [
    # ((17, 5), (0, 2)),
    # ((0, 9), (0, 1)),
    ((0, 5), (0, 2)), # Thumb abduction/ adduction
    ((0, 1), (1, 2)),
    ((1, 2), (2, 3)),
    ((2, 3), (3, 4)),

    ((0, 5), (5, 6)),
    ((5, 6), (6, 7)),
    ((6, 7), (7, 8)),

    ((0, 9), (9, 10)),
    ((9, 10), (10, 11)),
    ((10, 11), (11, 12)),

    ((0, 13), (13, 14)),
    ((13, 14), (14, 15)),
    ((14, 15), (15, 16)),

    ((0, 17), (17, 18)),
    ((17, 18), (18, 19)),
    ((18, 19), (19, 20)),

    ((5, 9), (9, 10)),
    # ((9, 10), (13, 14)),
    # ((13, 14), (17, 18)),
    # ((5, 8), (9, 12)),
    # ((9, 12), (13, 16)),
    # ((13, 16), (17, 20))
]


class ClassificationMethods:
    FLAT_COORDINATES = 'FLAT_COORDINATES'
    ANGLES = 'ANGLES'
    ANGLES_AND_FLAT_CO = 'ANGLES_AND_FLAT_COORDINATES'
    ENSEMBLE_1 = 'ENSEMBLE_1'
