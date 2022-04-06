from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import logging
import pygame

from src.utils.constants import EDGES_CUBE, VERTICES_CUBE
from src.pose_estimation.vertices_mapper import EDGES_MEDIA_PIPE, VERTICES_DEFAULT_MP
# from feature_extraction.pre_processor import get_angles, pre_process

from scipy.spatial import distance

LAST_VERTICES = [VERTICES_DEFAULT_MP]


def Cube():
    glBegin(GL_LINES)
    for edge in EDGES_CUBE:
        for vertex in edge:
            glVertex3fv(VERTICES_CUBE[vertex])
    glEnd()


def line_model(current_edges, current_vertices, color):
    glLineWidth(4.0)
    glBegin(GL_LINES)
    glColor3f(*color)
    for edge in current_edges:
        for vertex in edge:
            glVertex3fv(current_vertices[vertex])
    glEnd()


def joint_model(current_vertices, color):
    glEnable(GL_POINT_SMOOTH)
    glPointSize(8.0)
    glBegin(GL_POINTS)
    glColor3f(*color)
    for pt in current_vertices:
        glVertex3fv(pt)
    glEnd()


def reference_line(current_vertices):
    glBegin(GL_LINES)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3fv(current_vertices[0])
    glVertex3fv(current_vertices[9])
    glEnd()

def reference_line_y(current_vertices):
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 0.0)
    glVertex3fv(current_vertices[5])
    glVertex3fv(current_vertices[17])
    glEnd()


def draw_axes():
    glBegin(GL_LINES)
    # x red
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(10.0, 0.0, 0.0)

    # y light blue
    glColor3f(0.0, 0.5, 0.5)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 10.0, 0.0)

    # z purple
    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 10.0)
    glEnd()

def hand_model(vertices, current_edges, color=(1.0, 1.0, 1.0)):
    line_model(current_edges, vertices, color)
    joint_model(vertices, color)
    reference_line(vertices)
    reference_line_y(vertices)


def display_info(vertices):
    # drawText((-2, -1, 0), 'Thumb')
    # drawText((-1, -1, 0), 'Index')
    # drawText((-0, -1, 0), 'Middle')
    # drawText((1, -1, 0), 'Ring')
    # drawText((2, -1, 0), 'Pinky')
    #
    # angles = get_angles(vertices)
    # drawText((-2.5, -1.25, 0), 'Angle')
    # drawText((-2, -1.25, 0), str(angles[0]))
    # drawText((-1, -1.25, 0), str(angles[1]))
    # drawText((0, -1.25, 0), str(angles[2]))
    # drawText((1, -1.25, 0), str(angles[3]))
    # drawText((2, -1.25, 0), str(angles[4]))

    drawText((-2, -1.25, 0), str(distance.euclidean(vertices[5], vertices[17])))



def drawText(position, textString):
    font = pygame.font.Font(None, 20)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def render_static_and_dynamic(queue, land_mark):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    glRotatef(90, 0, 0, 0)

    while True:
        try:
            current_edges = EDGES_MEDIA_PIPE
            if not queue.empty():
                current_vertices = queue.get()
                LAST_VERTICES[0] = current_vertices

            else:
                last_vertices_local = LAST_VERTICES[0]
                current_vertices = last_vertices_local


        except Exception as e:
            logging.error('Error: {}'.format(e))
            current_vertices = VERTICES_CUBE
            current_edges = EDGES_CUBE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        current_vertices_2 = land_mark
        current_edges2 = EDGES_MEDIA_PIPE

        glRotatef(1, 0, 1, 0)
        # TODO: Remove this after pre-processing testing
        # current_vertices = pre_process(current_vertices)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        hand_model(current_vertices, current_edges)
        hand_model(current_vertices_2, current_edges2, color=(0.5, 0.5, 1.0))
        # display_info(current_vertices)
        draw_axes()
        pygame.display.flip()
        pygame.time.wait(10)

def render(queue):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    glRotatef(90, 0, 0, 0)

    while True:
        try:
            current_edges = EDGES_MEDIA_PIPE
            if not queue.empty():
                current_vertices, angles = queue.get()
                LAST_VERTICES[0] = current_vertices

            else:
                last_vertices_local = LAST_VERTICES[0]
                current_vertices = last_vertices_local
        except Exception as e:
            logging.error('Error: {}'.format(e))
            current_vertices = VERTICES_CUBE
            current_edges = EDGES_CUBE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 0, 1, 0)
        # TODO: Remove this after pre-processing testing
        # current_vertices = pre_process(current_vertices)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        hand_model(current_vertices, current_edges)
        display_info(current_vertices)
        draw_axes()
        pygame.display.flip()
        pygame.time.wait(10)

def render_static_2_hands(land_mark_1, land_mark_2):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    glRotatef(90, 0, 0, 0)

    while True:
        try:
            current_edges = EDGES_MEDIA_PIPE
            current_vertices_1 = land_mark_1
            current_vertices_2 = land_mark_2

        except Exception as e:
            logging.error('Error: {}'.format(e))
            current_vertices_1 = VERTICES_CUBE
            current_vertices_2 = VERTICES_CUBE
            current_edges = EDGES_CUBE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        hand_model(current_vertices_1, current_edges)
        hand_model(current_vertices_2, current_edges, color=(0.5, 0.5, 1.0))

        draw_axes()
        glRotatef(0.5, 0, 1, 0)
        pygame.display.flip()
        pygame.time.wait(10)


def render_static(land_mark):
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -5)

    glRotatef(90, 0, 0, 0)

    while True:
        try:
            current_edges = EDGES_MEDIA_PIPE
            current_vertices = land_mark

        except Exception as e:
            logging.error('Error: {}'.format(e))
            current_vertices = VERTICES_CUBE
            current_edges = EDGES_CUBE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        hand_model(current_vertices, current_edges)
        display_info(current_vertices)
        draw_axes()
        glRotatef(1, 0, 1, 0)
        pygame.display.flip()
        pygame.time.wait(10)
