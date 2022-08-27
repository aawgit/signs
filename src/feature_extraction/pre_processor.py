import logging
import math
import numpy as np
from scipy.spatial import distance

from src.utils.constants import EDGE_PAIRS_FOR_ANGLES, VERTICES_TO_IGNORE, IMPORTANT_FEATURES

logging.basicConfig(level=logging.INFO)


def get_sin_and_cos(a, b):
    c = math.sqrt(a ** 2 + b ** 2)
    cos = a / c
    sin = b / c
    return sin, cos


def rotate_pose(landmarks, rotation_matrix):
    rotated_vertices = []
    for frame_vertex in landmarks:
        point = np.array(frame_vertex)
        rotated = np.matmul(rotation_matrix, point)
        rotated_vertices.append((rotated[0], rotated[1], rotated[2]))
    return rotated_vertices


def stop_rotation_around_x(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]
    arr = np.array([frame_vertices[5], frame_vertices[13], frame_vertices[17]])
    point_2 = arr.sum(axis=0)

    a = point_2[1] - point_1[1]
    b = point_2[2] - point_1[2]

    sin, cos = get_sin_and_cos(a, b)
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, cos, sin],
                                  [0, -sin, cos]])

    rotated_vertices = rotate_pose(frame_vertices, rotation_matrix_x)
    return rotated_vertices, np.arcsin(sin)


def stop_rotation_around_y(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]

    arr = np.array([frame_vertices[5], frame_vertices[13], frame_vertices[17]])
    point_2 = arr.sum(axis=0)

    a = point_2[0] - point_1[0]
    b = point_2[2] - point_1[2]

    sin, cos = get_sin_and_cos(a, b)
    rotation_matrix_y = np.array([[cos, 0, sin],
                                  [0, 1, 0],
                                  [-sin, 0, cos]])
    rotated_vertices = rotate_pose(frame_vertices, rotation_matrix_y)
    return rotated_vertices, np.arcsin(sin)


def stop_rotation_around_z(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]
    point_2 = frame_vertices[reference_joints[1]]

    a = point_2[0] - point_1[0]
    b = point_2[1] - point_1[1]

    sin, cos = get_sin_and_cos(a, b)

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


def scale_vertices(frame_vertices, scale_factor_v=1):
    # frame_vertices = [(point[0] * 1000, point[1] * 1000, point[2] * 1000) for point in frame_vertices]

    wrist_joint = frame_vertices[0]
    middle_base_joint = frame_vertices[9]
    base_limb_length = distance.euclidean(wrist_joint, middle_base_joint)
    scale_ratio_v = scale_factor_v / base_limb_length
    base_vector_v = [middle_base_joint[i] - wrist_joint[i] for i in range(0, 3)]
    unit_vector_v = _unit_vector(base_vector_v)

    base_limb_length2 = distance.euclidean(frame_vertices[17], frame_vertices[5])
    scale_ratio_h = 0.7 / base_limb_length2
    unit_vector_h = _unit_vector([frame_vertices[17][i] - frame_vertices[5][i] for i in range(0, 3)])

    processed = []
    for land_mark in frame_vertices:
        processed.append([land_mark[0] * scale_ratio_h * unit_vector_h[0],
                          land_mark[1] * scale_ratio_v * unit_vector_v[1],
                          land_mark[2] * (scale_ratio_v + scale_ratio_h) / 2])
    # for land_mark in frame_vertices:
    #     processed.append([land_mark[0] * scale_ratio_v,
    #                       land_mark[1] * scale_ratio_v,
    #                       land_mark[2] * scale_ratio_v])
    return processed, None


def normalize_movement(frame_vertices, angles=None):
    origin_coordinates = frame_vertices[0]
    relocated_vertices = []
    for frame_vertex in frame_vertices:
        relocated_vertices.append((frame_vertex[0] - origin_coordinates[0], frame_vertex[1] - origin_coordinates[1],
                                   frame_vertex[2] - origin_coordinates[2]))
    return relocated_vertices, angles


def pre_process(land_marks):
    steps = [
        normalize_movement,
        stop_rotation_around_x,
        stop_rotation_around_z,
        stop_rotation_around_y,
        scale_vertices
    ]
    args_for_steps = [
        (),
        ([0, 9],),
        ([0, 9],),
        ([5, 17],),
        ()
    ]

    processed = land_marks
    rotations = []
    for idx, step in enumerate(steps):
        processed, rotation = step(processed, *args_for_steps[idx])
        if rotation: rotations.append(math.degrees(rotation))
    return tuple(processed), rotations


def get_angle(v1, v2):
    # TODO: Check rotations implementation
    # Check rotation reference lines
    # Check how angles>90
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


def pre_process_single_frame(land_mark):
    if not land_mark: return
    current_vertices, angles = pre_process(land_mark)
    return current_vertices, angles


def run_pre_process_steps(pose_q, processed_q_1, duplicate_queues=None, for_labelling=False):
    logging.info('Pre-processing...')
    while True:
        try:
            if not pose_q.empty():
                landmarks, frame_no = pose_q.get()
                landmarks, original_angles = build_feature_vector(landmarks)
                processed_q_1.put((landmarks, original_angles))
                if duplicate_queues:
                    for q in duplicate_queues:
                        if for_labelling:
                            q.put(((landmarks, original_angles), frame_no))
                        else:
                            q.put((landmarks, original_angles))
            else:
                pass
        except Exception as e:
            logging.error(e)
            break


def flatten(landmark, decimal_points=4):
    lm_row = []
    for landmark_point in landmark:
        lm_row.extend(np.round(landmark_point, decimal_points))
    return lm_row


def get_angles(landmarks):
    angles = []
    for limb_pair in EDGE_PAIRS_FOR_ANGLES:
        limb2 = [landmarks[limb_pair[1][1]][i] - landmarks[limb_pair[1][0]][i] for i in range(0, 3)]
        limb1 = [landmarks[limb_pair[0][1]][i] - landmarks[limb_pair[0][0]][i] for i in range(0, 3)]
        angle = get_angle(limb2, limb1)
        angles.append((angle - 90) / 90)

    return angles


def select_important_features(flat_co_and_angles):
    selected = []
    for ft in IMPORTANT_FEATURES:
        selected.append(flat_co_and_angles[ft])
    return selected


def build_feature_vector(landmarks, ratio=1):
    preprocessed_landmark, rotations = pre_process(landmarks)
    angles = get_angles(preprocessed_landmark)
    if ratio != 1:
        angles = [angle * ratio for angle in angles]
    preprocessed_landmark = list(preprocessed_landmark)
    for index in sorted(VERTICES_TO_IGNORE, reverse=True):
        del preprocessed_landmark[index]

    flatted = flatten(preprocessed_landmark)
    flatted.extend(angles)
    flatted = select_important_features(flatted)
    return flatted, rotations
