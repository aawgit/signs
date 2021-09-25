from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import logging
import pygame

from utils.constants import EDGES_MEDIA_PIPE, EDGES_CUBE, VERTICES_DEFAULT_MP, VERTICES_CUBE
from feature_extraction.pre_processor import get_angles

LAST_VERTICES = [VERTICES_DEFAULT_MP]


def Cube():
    glBegin(GL_LINES)
    for edge in EDGES_CUBE:
        for vertex in edge:
            glVertex3fv(VERTICES_CUBE[vertex])
    glEnd()


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


def hand_model(vertices, current_edges):
    line_model(current_edges, vertices)
    joint_model(vertices)
    reference_line(vertices)


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

        # glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        hand_model(current_vertices, current_edges)
        display_info(current_vertices)
        pygame.display.flip()
        pygame.time.wait(10)
