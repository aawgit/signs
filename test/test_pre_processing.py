import pytest
import math

from src.feature_extraction.pre_processor import get_angle, normalize_movement, stop_rotation_around_x, \
    scale_vertices, stop_rotation_around_z, stop_rotation_around_y
from test.dummies import pre_processing_out, raw_landmark, movements_removed_correct, rot_x_rem_correct

dummy_landmark = [(0.0, 0.0, 0.0),
                  (0.494931877409792, 0.06352074855624598, 0.34365336068146485),
                  (0.8099269739011388, 0.15019027958648754, 0.3863764099575968),
                  (1.0627734165622655, 0.17654285474865875, 0.3351322967874831),
                  (1.3355245591151363, 0.1746283386709087, 0.2561532547283659),
                  (0.7251434824494394, 0.6641893493676517, 0.18171251483504158),
                  (0.932902424851623, 0.9301016554557303, 0.06684066416542424),
                  (1.111041704477273, 1.0493965499714752, 0.008893074079888335),
                  (1.264307445978213, 1.1076542445867825, -0.04771070153657935),
                  (0.607294174410951, 0.6822616051880965, -0.09484187247584792),
                  (0.8268744701786342, 0.9467214854650475, -0.2510124345111621),
                  (1.0304768385086644, 1.0758103060839097, -0.3722329279343161),
                  (1.2390680656671733, 1.1729601836348562, -0.44564335326787186),
                  (0.4920770151005568, 0.6442160074341488, -0.35481796318674524),
                  (0.7176113862331885, 0.8816557935631295, -0.5372253291145253),
                  (0.9246134009031519, 1.0115041532962357, -0.6215840979328423),
                  (1.1160672470056043, 1.1305891424379066, -0.6914260697313789),
                  (0.37608141524413, 0.5464045797705996, -0.5721712164066598),
                  (0.5518376222461113, 0.7192377798276867, -0.7531665396810316),
                  (0.712010096144453, 0.8550425776047149, -0.8452783773753918),
                  (0.8582676344036527, 0.9898679827714314, -0.9188083667001632)]


def test_get_angle_v2():
    line1 = ((0, 0), (1, 1))
    line2 = ((1, 1), (0, 1))

    v1 = [line1[1][i] - line1[0][i] for i in range(0, 2)]
    v2 = [line2[1][i] - line2[0][i] for i in range(0, 2)]
    angle = get_angle(v1, v2)
    assert angle == 45

    line1 = ((0, 0, 0), (0, 1, 1))
    line2 = ((0, 1, 1), (0, 2, 0))

    v1 = [line1[1][i] - line1[0][i] for i in range(0, 3)]
    v2 = [line2[1][i] - line2[0][i] for i in range(0, 3)]
    angle = get_angle(v1, v2)
    assert angle == 90

    line1 = ((0, 0, 0), (1, 0, 0))
    line2 = ((0, 0, 0), (-1, 1, 0))

    # v1 = [line1[1][i] - line1[0][i] for i in range(0, 3)]
    # v2 = [line2[1][i] - line2[0][i] for i in range(0, 3)]
    # angle = get_angle_v2(v1, v2)
    # assert angle == -90


def test_normalize_movement():
    movement_removed = normalize_movement(raw_landmark)
    assert movement_removed == (movements_removed_correct, None)


def test_stop_rotation_around_x():
    rot_x_removed = stop_rotation_around_x(movements_removed_correct, [0, 9])
    assert rot_x_removed == rot_x_rem_correct


def test_run_preprocessing_steps():
    pre_processed, other = scale_vertices(raw_landmark)
    pre_processed, other = normalize_movement(pre_processed)
    pre_processed, other = stop_rotation_around_x(pre_processed, [0, 9])
    pre_processed, other = stop_rotation_around_z(pre_processed, [0, 9])
    pre_processed, other = stop_rotation_around_y(pre_processed, [5, 17])
    assert pre_processed == pre_processing_out


def test_pre_process():
    steps = [
        normalize_movement,
        stop_rotation_around_x,
        stop_rotation_around_z,
        stop_rotation_around_y,
        scale_vertices,
    ]
    args_for_steps = [
        (),
        ([0, 9],),
        ([0, 9],),
        ([5, 17],),
        ()
    ]
    processed = dummy_landmark
    rotations = []
    for idx, step in enumerate(steps):
        processed, rotation = step(processed, *args_for_steps[idx])
        if rotation: rotations.append(math.degrees(rotation))
    result_1 = tuple(processed), rotations

    steps = [
        scale_vertices,
        normalize_movement,
        stop_rotation_around_x,
        stop_rotation_around_z,
        stop_rotation_around_y,
    ]
    args_for_steps = [
        (),
        (),
        ([0, 9],),
        ([0, 9],),
        ([5, 17],),

    ]
    processed = dummy_landmark
    rotations = []
    for idx, step in enumerate(steps):
        processed, rotation = step(processed, *args_for_steps[idx])
        if rotation: rotations.append(math.degrees(rotation))
    result_2 = tuple(processed), rotations

    assert result_1 == result_2
