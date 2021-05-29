def scale_vertices():
    pass


def normalize_movement(frame_vertices):
    origin_coordinates = frame_vertices[0]
    relocated_vertices = []
    for frame_vertex in frame_vertices:
        relocated_vertices.append((frame_vertex[0] - origin_coordinates[0], frame_vertex[1] - origin_coordinates[1],
                                   frame_vertex[2] - origin_coordinates[2]))
    return relocated_vertices


def pre_process(land_marks):
    scale_factor = 4
    processed = []
    for land_mark in land_marks:
        processed.append((land_mark.x * scale_factor, land_mark.y * scale_factor * (-1), land_mark.z * scale_factor))
    processed = normalize_movement(processed)
    return tuple(processed)
