import logging
import math
import numpy as np
import pandas as pd
from scipy.spatial import distance

logging.basicConfig(level=logging.INFO)


def stop_rotation_around_x(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]
    # point_2 = frame_vertices[reference_joints[1]]
    arr = np.array([frame_vertices[5], frame_vertices[13], frame_vertices[17]])
    point_2 = arr.sum(axis=0)

    a = point_2[1] - point_1[1]
    b = point_2[2] - point_1[2]
    c = math.sqrt(a ** 2 + b ** 2)
    cos = a / c
    sin = b / c
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cos, sin],
                                  [0, -sin, cos]])

    rotation_matrix_x_2 = np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]])

    rotated_vertices = []
    for frame_vertex in frame_vertices:
        point = np.array(frame_vertex)
        rotated = np.matmul(rotation_matrix_x, point)
        # rotated = np.matmul(rotation_matrix_x_2, rotated)
        rotated_vertices.append((rotated[0], rotated[1], rotated[2]))
    return rotated_vertices, np.arcsin(sin)


def stop_rotation_around_y(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]
    # point_2 = frame_vertices[reference_joints[1]]

    arr = np.array([frame_vertices[5], frame_vertices[13], frame_vertices[17]])
    point_2 = arr.sum(axis=0)

    a = point_2[0] - point_1[0]
    b = point_2[2] - point_1[2]
    c = math.sqrt(a ** 2 + b ** 2)
    cos = a / c
    sin = b / c
    rotation_matrix_y = np.array([[cos, 0, sin],
                                  [0, 1, 0],
                                  [-sin, 0, cos]])

    rotation_matrix_y_2 = np.array([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]])

    rotated_vertices = []
    for frame_vertex in frame_vertices:
        point = np.array(frame_vertex)
        rotated = np.matmul(rotation_matrix_y, point)
        # rotated = np.matmul(rotation_matrix_y_2, rotated)
        rotated_vertices.append((rotated[0], rotated[1], rotated[2]))
    return rotated_vertices, np.arcsin(sin)


def stop_rotation_around_z(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]
    point_2 = frame_vertices[reference_joints[1]]

    a = point_2[0] - point_1[0]
    b = point_2[1] - point_1[1]

    c = math.sqrt(a ** 2 + b ** 2)
    cos = a / c
    sin = b / c

    rotation_matrix_z = np.array([[cos, sin, 0],
                                  [-sin, cos, 0],
                                  [0, 0, 1]])

    rotation_matrix_z_2 = np.array([[0, -1, 0],
                                    [1, 0, 0],
                                    [0, 0, 1]])

    rotated_vertices = []
    for frame_vertex in frame_vertices:
        point = np.array(frame_vertex)
        rotated = np.matmul(rotation_matrix_z, point)
        rotated = np.matmul(rotation_matrix_z_2, rotated)
        rotated_vertices.append((rotated[0], rotated[1], rotated[2]))
    return rotated_vertices, np.arcsin(sin)


def scale_vertices2(frame_vertices, scale_factor_v=1):
    wrist_joint = frame_vertices[0]
    index_base_joint = frame_vertices[9]
    base_limb_length = distance.euclidean(wrist_joint, index_base_joint)
    scale_ratio_v = scale_factor_v / base_limb_length
    base_vector_v = [index_base_joint[i] - wrist_joint[i] for i in range(0, 3)]
    unit_vector_v = _unit_vector(base_vector_v)

    base_limb_length2 = distance.euclidean(frame_vertices[17], frame_vertices[5])
    scale_ratio_h = 0.7 / base_limb_length2
    unit_vector_h = _unit_vector([frame_vertices[17][i] - frame_vertices[5][i] for i in range(0, 3)])

    processed = []
    for land_mark in frame_vertices:
        processed.append([land_mark[0] * scale_ratio_h * unit_vector_h[0],
                          land_mark[1] * scale_ratio_v * unit_vector_v[1],
                          land_mark[2] * (scale_ratio_v + scale_ratio_h) / 2])
    return processed, None


def scale_vertices(frame_vertices, scale_factor=1):
    wrist_joint = frame_vertices[0]
    index_base_joint = frame_vertices[9]
    base_limb_length = distance.euclidean(wrist_joint, index_base_joint)
    scale_factor = scale_factor / base_limb_length
    processed = []
    for land_mark in frame_vertices:
        processed.append((land_mark[0] * scale_factor, land_mark[1] * scale_factor, land_mark[2] * scale_factor))
    return processed


def normalize_movement(frame_vertices):
    origin_coordinates = frame_vertices[0]
    relocated_vertices = []
    for frame_vertex in frame_vertices:
        relocated_vertices.append((frame_vertex[0] - origin_coordinates[0], frame_vertex[1] - origin_coordinates[1],
                                   frame_vertex[2] - origin_coordinates[2]))
    return relocated_vertices, None


def pre_process(land_marks, steps, args_for_steps):
    processed = land_marks
    rotations = []
    for idx, step in enumerate(steps):
        processed, rotation = step(processed, *args_for_steps[idx])
        if rotation: rotations.append(math.degrees(rotation))
    return tuple(processed), rotations


def get_angle_v2(v1, v2):
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    angle = angle_between(v1, v2)
    return math.degrees(np.pi - angle)


def flatten_points(land_marks: list):
    flatten_coordinates = []
    for point in land_marks:
        for coordinate in point:
            flatten_coordinates.append(coordinate)
    return flatten_coordinates


def un_flatten_points(flatten_coordinates: list):
    landmark_points = []
    for i in range(0, len(flatten_coordinates), 3):
        landmark_points.append(flatten_coordinates[i:i + 3])
    return landmark_points


def _unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def normalize_flat_coordinates_scale(land_marks: pd.DataFrame):
    land_marks_wo_sign = land_marks.drop('sign', axis=1)
    limb_size_adjusted_landmarks = []
    for index, row in land_marks_wo_sign.iterrows():
        flattened_row = list(un_flatten_points(list(row)))
        normalized_row = scale_vertices2(flattened_row)
        limb_size_adjusted_landmarks.append(flatten_points(normalized_row))
    adjusted_df_without_sign = pd.DataFrame(limb_size_adjusted_landmarks, columns=land_marks_wo_sign.columns.values)
    adjusted_df_without_sign['sign'] = land_marks['sign']
    return adjusted_df_without_sign


def normalize_limb_sizes(land_mark):
    reference = [
        ((0, 1), 46.22 * 1.1 / 2),
        ((0, 9), 68.12 * 1.1),
        ((0, 5), 64.60 * 1.1),
        ((0, 13), 58.0 * 1.1),
        ((0, 17), 53.69 * 1.1),

        # ((9, 1), 40),
        ((9, 5), 20),
        ((9, 13), 20),
        ((9, 17), 40),

        ((1, 2), 46.22 * 1.1 / 2),
        ((2, 3), 31.57),
        ((3, 4), 21.67),

        ((5, 6), 39.78),
        ((6, 7), 22.38),
        ((7, 8), 15.82),

        ((9, 10), 44.63),
        ((10, 11), 26.33),
        ((11, 12), 17.4),

        ((13, 14), 41.37),
        ((14, 15), 25.65),
        ((15, 16), 17.30),

        ((17, 18), 32.74),
        ((18, 19), 18.11),
        ((19, 20), 15.96)
    ]

    limb_adjusted_lm = [(0, 0, 0) for _ in range(0, 21)]

    for reference_limb_sizes in reference:
        reference_limb_joints = reference_limb_sizes[0]
        reference_size = reference_limb_sizes[1]
        limb_start_joint = reference_limb_joints[0]
        limb_end_joint = reference_limb_joints[1]
        actual_limb = [land_mark[limb_end_joint][i] - land_mark[limb_start_joint][i] for i in range(0, 3)]
        new_limb = [point * reference_size / 70 for point in _unit_vector(actual_limb)]
        new_limb_end = [limb_adjusted_lm[limb_start_joint][i] + new_limb[i] for i in range(0, 3)]
        limb_adjusted_lm[limb_end_joint] = new_limb_end

    return limb_adjusted_lm


def pre_process_single_frame(land_mark):
    if not land_mark: return
    steps = [
        normalize_movement,
        stop_rotation_around_x,
        stop_rotation_around_z,
        stop_rotation_around_y,
        scale_vertices2,
        # normalize_limb_sizes
    ]
    args_for_steps = [
        (),
        ([0, 9],),
        ([0, 9],),
        ([5, 17],),
        (),
        # ()
    ]
    current_vertices, angles = pre_process(land_mark, steps, args_for_steps)
    # coordinates = []
    # for current_vertex in current_vertices:
    #     for coordinate in current_vertex:
    #         coordinates.append(np.round(coordinate, 8))
    # logging.info(coordinates)
    return current_vertices, angles


def run_pre_process_steps(pose_q, processed_q_1, duplicate_queues=None, for_labelling=False):
    logging.info('Pre-processing...')
    steps = [
        normalize_movement,
        stop_rotation_around_x,
        stop_rotation_around_z,
        stop_rotation_around_y,
        scale_vertices2,
    ]
    args_for_steps = [(), ([0, 9],), ([0, 9],), ([5, 17],), ()]
    while True:
        try:
            if not pose_q.empty():
                current_vertices, frame_no = pose_q.get()
                current_vertices = pre_process(current_vertices, steps, args_for_steps)
                # coordinates = []
                # for current_vertex in current_vertices:
                #     for coordinate in current_vertex:
                #         coordinates.append(np.round(coordinate, 8))
                # logging.info(coordinates)
                processed_q_1.put(current_vertices)
                if duplicate_queues:
                    for q in duplicate_queues:
                        if for_labelling:
                            q.put((current_vertices, frame_no))
                        else:
                            q.put(current_vertices)
            else:
                pass
        except Exception as e:
            logging.error(e)
            break

# References for normalizing
# S. Agahian et al. / Engineering Science and Technology, an International Journal 23 (2020) 196–203
