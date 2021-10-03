import logging
import math
import numpy as np
from scipy.spatial import distance

def stop_rotation_around_x(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]
    point_2 = frame_vertices[reference_joints[1]]
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
    return rotated_vertices


def stop_rotation_around_y(frame_vertices, reference_joints):
    point_1 = frame_vertices[reference_joints[0]]
    point_2 = frame_vertices[reference_joints[1]]

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
    return rotated_vertices


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
    return rotated_vertices


def scale_vertices(frame_vertices, scale_factor):
    # TODO: This isn't correct. Each limb should be increased/decreased using a unit vector * change in base limb length
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
    return relocated_vertices


def pre_process(land_marks):
    scale_factor = 1
    # TODO: Change to below method after rotation testing
    processed = land_marks

    processed = normalize_movement(processed)
    processed = scale_vertices(processed, scale_factor)


    processed = stop_rotation_around_x(processed, [0, 9])

    processed = stop_rotation_around_z(processed, [0, 9])

    processed = stop_rotation_around_y(processed, [5, 17])
    return tuple(processed)


def angle(v1, v2, acute=True):
    v1 = [v1[1][0] - v1[0][0], v1[1][1] - v1[0][1], v1[1][2] - v1[0][2]]
    v2 = [v2[1][0] - v2[0][0], v2[1][1] - v2[0][1], v2[1][2] - v2[0][2]]
    # v1 is your firsr vector
    # v2 is your second vector
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if (acute == True):
        angle = angle
    else:
        angle = 2 * np.pi - angle
    return np.round(180 * angle / np.pi)


def get_angles(frame_vertices):
    thumb_angle = angle((frame_vertices[2], frame_vertices[0]), (frame_vertices[4], frame_vertices[2]))
    index_angle = angle((frame_vertices[5], frame_vertices[0]), (frame_vertices[8], frame_vertices[5]))
    middle_angle = angle((frame_vertices[9], frame_vertices[0]), (frame_vertices[12], frame_vertices[9]))
    ring_angle = angle((frame_vertices[13], frame_vertices[0]), (frame_vertices[16], frame_vertices[13]))
    pinky_angle = angle((frame_vertices[17], frame_vertices[0]), (frame_vertices[20], frame_vertices[9]))
    return [thumb_angle, index_angle, middle_angle, ring_angle, pinky_angle]


def run_pre_process_steps(pose_q, processed_q_1, processed_q_2, for_labelling=False):
    while True:
        try:
            if not pose_q.empty():
                current_vertices, frame_no = pose_q.get()
                current_vertices = pre_process(current_vertices)
                processed_q_1.put(current_vertices)
                if for_labelling:
                    processed_q_2.put((current_vertices, frame_no))
            else:
                pass
        except Exception as e:
            logging.error(e)
            break