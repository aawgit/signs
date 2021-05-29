import logging
import multiprocessing
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import queue

from points_generator import dynamic_images
from pre_processor import pre_process

vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
)


vertices_2 = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 2),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 2),
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
)

edges = (
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

edges_2 = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20)
)

data_queue = queue.Queue()


# data_queue.put(vertices)
# data_queue.put(vertices_1)

def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


LAST_VERTICES = [vertices_2]


def model_1(data_queue):
    glBegin(GL_LINES)
    current_edges = edges_2
    last_vertices_local = LAST_VERTICES[0]
    try:
        if not data_queue.empty():
            current_vertices = data_queue.get()
            current_vertices = pre_process(current_vertices)
            LAST_VERTICES[0] = current_vertices

        else:
            current_vertices = last_vertices_local
    except Exception as e:
        logging.error('Error: {}'.format(e))
        current_vertices = vertices
        current_edges = edges

    for edge in current_edges:
        for vertex in edge:
            glVertex3fv(current_vertices[vertex])
    glEnd()


def render(queue):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        model_1(queue)
        # Cube()
        pygame.display.flip()
        pygame.time.wait(10)



def worker(name, que):
    dynamic_images(que)
    # que.put("%d is done" % name)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=3)
    m = multiprocessing.Manager()
    q = m.Queue()
    workers = pool.apply_async(worker, (33, q))
    render(q)
