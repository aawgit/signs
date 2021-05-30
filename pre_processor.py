import math
import numpy as np

def rotate_around_axis_x(frame_vertices):
    reference_joints = [0, 9]
    point_1 = frame_vertices[reference_joints[0]]
    point_2 = frame_vertices[reference_joints[1]]
    a = point_2[1] - point_1[1]
    b = point_2[2] - point_1[2]
    c = math.sqrt(a**2 + b**2)
    cos = b/c
    sin = a/c
    print(np.rad2deg(np.arccos(cos)))
    rotation_matrix_x = np.array([[1, 0, 0],
                                [0, cos, -sin],
                                [0, sin, cos]])

    rotation_matrix_x_2 = np.array([[1, 0, 0],
                                  [0, 0, 1],
                                  [0, -1, 0]])

    rotated_vertices = []
    for frame_vertex in frame_vertices:
        point = np.array(frame_vertex)
        rotated = np.matmul(rotation_matrix_x, point)
        rotated = np.matmul(rotation_matrix_x_2, rotated)
        rotated_vertices.append((rotated[0], rotated[1], rotated[2]))
    return rotated_vertices
    # return frame_vertices


def rotate_around_axis_z(frame_vertices):
    reference_joints = [0, 9]
    point_1 = frame_vertices[reference_joints[0]]
    point_2 = frame_vertices[reference_joints[1]]
    a = point_2[1] - point_1[1]
    b = point_2[0] - point_1[0]
    c = math.sqrt(a**2 + b**2)
    cos = b/c
    sin = a/c

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
    wrist_joint = frame_vertices[0]
    index_base_joint = frame_vertices[5]
    base_limb_length = (index_base_joint[0] - wrist_joint[0])**2 + (index_base_joint[1] - wrist_joint[1])**2 + (index_base_joint[2] - wrist_joint[2])**2
    base_limb_length = math.sqrt(base_limb_length)
    scale_factor = scale_factor/base_limb_length
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
    processed = [(land_mark.x, land_mark.y * (-1), land_mark.z ) for land_mark in land_marks]
    # for land_mark in land_marks:
    #     processed.append((land_mark.x * scale_factor, land_mark.y * scale_factor * (-1), land_mark.z * scale_factor))
    processed = normalize_movement(processed)
    processed = scale_vertices(processed, scale_factor)
    processed = rotate_around_axis_z(processed)
    processed = rotate_around_axis_x(processed)
    return tuple(processed)
