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
                current_vertices, original_angles = pre_process(current_vertices, steps, args_for_steps)
                # coordinates = []
                # for current_vertex in current_vertices:
                #     for coordinate in current_vertex:
                #         coordinates.append(np.round(coordinate, 8))
                # logging.info(coordinates)
                processed_q_1.put((current_vertices, original_angles))
                if duplicate_queues:
                    for q in duplicate_queues:
                        if for_labelling:
                            q.put(((current_vertices, original_angles), frame_no))
                        else:
                            q.put((current_vertices, original_angles))
            else:
                pass
        except Exception as e:
            logging.error(e)
            break

def make_right_handed():
    l = [-0.29639709580602, 0.343219214739541, 0.254201292611363, -0.51895809688473, 0.744826255300432,
         0.49000753946639, -0.718621539306147, 1.09385266684722, 0.682628863301124, -0.913062629474884,
         1.36818884007564, 0.735345129854851, -0.273342832333906, 1.03462806361304, 0.195871885362095,
         -0.211616846618731, 1.28513751923693, 0.488082013541652, -0.208794318593136, 1.08958852067422,
         0.402876843606042, -0.216737444022668, 0.941882385680188, 0.269714616263431, -0.081351446133997,
         0.972028704152652, 0.114832296412129, -0.056451457669432, 1.22680382086875, 0.476175728772469,
         -0.070528201032265, 0.968522024327624, 0.413014049182185, -0.053368300059968, 0.81946256724041,
         0.292803262967062, 0.126206667156132, 0.894176662733648, 0.033072516787178, 0.092516038228765,
         1.09202948941252, 0.438226301612747, 0.068732122174872, 0.781937620386251, 0.382612889423777, 0.08152040904955,
         0.617595405623193, 0.283732443368985, 0.338949180810885, 0.86170735551817, -0.033072516787178,
         0.354526848480082, 1.24057640313612, 0.31402207856927, 0.395487890616822, 1.47202837548277, 0.519868002371577,
         0.457489914270733, 1.73876901970362, 0.697934426047106, 6

         ]

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

    l_f = un_flatten_points(l)
    n = 2
    # l2_f =[ [x if (i % n or i==0) else -x for i, x in
    l2_f = [[-x if (i == 0) else x for i, x in enumerate(coordinate)] for coordinate in l_f]
    l2 = flatten_points(l2_f)
    print(l2)
# References for normalizing
# S. Agahian et al. / Engineering Science and Technology, an International Journal 23 (2020) 196–203
