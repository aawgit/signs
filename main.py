import logging
import multiprocessing
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import queue

from points_generator import dynamic_images
from pre_processor import pre_process, get_angles

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

def line_model(current_edges, current_vertices):
    glLineWidth(4.0)
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)
    for edge in current_edges:
        for vertex in edge:
            glVertex3fv(current_vertices[vertex])
    glEnd()

def joint_model(current_vertices):
    glEnable(GL_POINT_SMOOTH)
    glPointSize(8.0)
    glBegin(GL_POINTS)
    glColor3f(1.0, 1.0, 1.0)
    for pt in current_vertices:
        glVertex3fv(pt)
    glEnd()

def reference_line(current_vertices):
    glBegin(GL_LINES)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3fv(current_vertices[0])
    glVertex3fv(current_vertices[9])
    glEnd()

def model_1(data_queue):
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
    line_model(current_edges, current_vertices)
    joint_model(current_vertices)
    reference_line(current_vertices)
    return current_vertices


def display_info(vertices):
    drawText((-2, -1, 0), 'Thumb')
    drawText((-1, -1, 0), 'Index')
    drawText((-0, -1, 0), 'Middle')
    drawText((1, -1, 0), 'Ring')
    drawText((2, -1, 0), 'Pinky')

    angles = get_angles(vertices)
    drawText((-2.5, -1.25, 0), 'Angle')
    drawText((-2, -1.25, 0), str(angles[0]))
    drawText((-1, -1.25, 0), str(angles[1]))
    drawText((0, -1.25, 0), str(angles[2]))
    drawText((1, -1.25, 0), str(angles[3]))
    drawText((2, -1.25, 0), str(angles[4]))


def drawText(position, textString):
    font = pygame.font.Font(None, 20)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)


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
        vertices = model_1(queue)
        # model_2()
        # Cube()
        display_info(vertices)
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
