from scipy.spatial import distance

fingers = [
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20]
]


def adjust_finger_bases(vertices):
    line_length = distance.euclidean(vertices[5], vertices[17])
    unit_vector = [
        (vertices[5][0] - vertices[17][0]) / line_length,
        (vertices[5][1] - vertices[17][1]) / line_length,
        (vertices[5][2] - vertices[17][2]) / line_length
    ]

    mid_point = [
        (vertices[5][0] + vertices[17][0]) / 2,
        (vertices[5][1] + vertices[17][1]) / 2,
        (vertices[5][2] + vertices[17][2]) / 2
    ]

    base_limb_length = distance.euclidean(vertices[0], vertices[9])
    scale_factor = 0.4 / base_limb_length
    model_finger_bases = [
        [mid_point[i]*unit_vector[i]*j*scale_factor for i in range(0, 3)] for j in [-.75, -.25, .25, .75]
    ]

    for f in range(0, 4):
        new_base = model_finger_bases[f]
        current_base = vertices[fingers[f][0]]
        difference = [new_base[0]-current_base[0], new_base[1]-current_base[1], new_base[2]-current_base[2]]
        for point in fingers[f]:
            vertices[point] = (vertices[point][0] + difference[0],
                               vertices[point][1] + difference[1],
                               vertices[point][2] + difference[2])

    return vertices


