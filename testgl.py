#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math

import numpy

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


def perspective_matrix(fovy, aspect, near, far):
    # Math from
    # https://www.opengl.org/discussion_boards/showthread.php/197893-View-and-Perspective-matrices
    #
    # I should understand this
    #
    # aspect is width/height

    s = 1.0/math.tan(fovy/2.0)
    sx, sy = s / aspect, s
    zz = (far+near)/(near-far)
    zw = 2*far*near/(near-far)
    return numpy.matrix([[sx, 0, 0, 0],
                         [0, sy, 0, 0],
                         [0, 0, zz, zw],
                         [0, 0, -1, 0]], dtype=numpy.float32).T

 
def camera_posrot(x, y, z, theta, phi):
    theta -= math.pi/2.
    if (theta >  math.pi/2): theta =  math.pi/2.
    if (theta < -math.pi/2): theta = -math.pi/2.
    if (phi > 2.*math.pi): phi -= 2.*math.pi
    if (phi < 0.): phi += 2.*math.pi
    ct = math.cos(theta)
    st = math.sin(theta)
    cp = math.cos(phi)
    sp = math.sin(phi)
    matrix = numpy.matrix([[    cp   ,   0.  ,   sp  , -x ],
                           [ -sp*st  ,  ct   , cp*st , -y ],
                           [ -sp*ct  , -st   , cp*ct , -z ],
                           [    0.   ,   0.  ,   0.  ,  1.]], dtype=numpy.float32)
    return matrix


# ======================================================================

glutInit(len(sys.argv), sys.argv)
glutInitContextVersion(3, 3)
glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
glutInitContextProfile(GLUT_CORE_PROFILE)
glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

glutInitWindowPosition(-1, -1)
glutInitWindowSize(600, 600)
window = glutCreateWindow("BLAH")
glutSetWindow(window)

vertex_shader = """
#version 330

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;
uniform mat3 model_normal;

layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Normal;
out vec3 aNormal;

void main(void)
{
  // gl_Position =  projection * view * model * in_Position;
  // aNormal = model_normal * in_Normal;
  gl_Position = projection * view * in_Position;
  aNormal = in_Normal;
}"""

fragment_shader = """
#version 330

uniform vec3 ambientcolor;
uniform vec3 light1color;
uniform vec3 light1dir;
uniform vec3 light2color;
uniform vec3 light2dir;
uniform vec4 color;

in vec3 aNormal;
out vec4 out_Color;

void main(void)
{
  vec3 norm = normalize(aNormal);
  vec3 diff1 = max(dot(norm, light1dir), 0.) * light1color;
  vec3 diff2 = max(dot(norm, light2dir), 0.) * light2color;
  vec3 col = (ambientcolor + diff1 + diff2) * vec3(color);
  out_Color = vec4(col, color[3]);
}"""

vtxshdrid = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vtxshdrid, vertex_shader)
glCompileShader(vtxshdrid)
if glGetShaderiv(vtxshdrid, GL_COMPILE_STATUS) == GL_TRUE:
    sys.stderr.write("Vertex shader compilation successful.\n")
else:
    sys.stderr.write("Vertex shader compilation error: {}\n"
                     .format(glGetShaderInfoLog(vtxshdrid)))

    sys.exit(-1)
    
fragshdrid = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragshdrid, fragment_shader)
glCompileShader(fragshdrid)
if glGetShaderiv(fragshdrid, GL_COMPILE_STATUS) == GL_TRUE:
    sys.stderr.write("Fragment shader compilation succesful.\n")
else:
    sys.stderr.write("Fragment shader compilation error: {}\n"
                     .format(glGetShaderInfoLog(fragshdrid)))
    sys.exit(-1)
    
progid = glCreateProgram()
glAttachShader(progid, vtxshdrid)
glAttachShader(progid, fragshdrid)
glLinkProgram(progid)

if glGetProgramiv(progid, GL_LINK_STATUS) != GL_TRUE:
    sys.stderr.write("{}\n".format(glGetProgramInfoLog(progid)))
    sys.exit(-1)

glUseProgram(progid)

loc = glGetUniformLocation(progid, "ambientcolor")
glUniform3fv(loc, 1, numpy.array([0.2, 0.2, 0.2]))
loc = glGetUniformLocation(progid, "light1color")
glUniform3fv(loc, 1, numpy.array([0.8, 0.8, 0.8]))
loc = glGetUniformLocation(progid, "light1dir")
glUniform3fv(loc, 1, numpy.array([0.22, 0.44, 0.88]))
loc = glGetUniformLocation(progid, "light2color")
glUniform3fv(loc, 1, numpy.array([0.3, 0.3, 0.3]))
loc = glGetUniformLocation(progid, "light2dir")
glUniform3fv(loc, 1, numpy.array([-0.88, -0.22, -0.44]))

loc = glGetUniformLocation(progid, "color")
glUniform4fv(loc, 1, numpy.array([1., 0.5, 0.5, 1.]))

matrix = perspective_matrix(1., 1., 0.1, 1000.)
loc = glGetUniformLocation(progid, "projection")
glUniformMatrix4fv(loc, 1, GL_FALSE, matrix)

matrix = camera_posrot(0., 0., 2., math.pi/4., math.pi/4.)
loc = glGetUniformLocation(progid, "view")
glUniformMatrix4fv(loc, 1, GL_FALSE, matrix.T)

box_num_triangles = 12

box_vertices = numpy.array( [ -0.5, -0.5,  0.5, 1.,
                              -0.5, -0.5, -0.5, 1.,
                              0.5, -0.5,  0.5, 1.,
                              0.5, -0.5,  0.5, 1.,
                              -0.5, -0.5, -0.5, 1.,
                              0.5, -0.5, -0.5, 1.,

                              -0.5,  0.5,  0.5, 1.,
                              0.5,  0.5,  0.5, 1.,
                              -0.5,  0.5, -0.5, 1.,
                              -0.5,  0.5, -0.5, 1.,
                              0.5,  0.5,  0.5, 1.,
                              0.5,  0.5, -0.5, 1.,

                              -0.5, -0.5, -0.5, 1.,
                              -0.5, -0.5,  0.5, 1.,
                              -0.5,  0.5, -0.5, 1.,
                              -0.5,  0.5, -0.5, 1.,
                              -0.5, -0.5,  0.5, 1.,
                              -0.5,  0.5,  0.5, 1.,

                              0.5,  0.5, -0.5, 1.,
                              0.5,  0.5,  0.5, 1.,
                              0.5, -0.5, -0.5, 1.,
                              0.5, -0.5, -0.5, 1.,
                              0.5,  0.5,  0.5, 1.,
                              0.5, -0.5,  0.5, 1.,

                              -0.5, -0.5,  0.5, 1.,
                              0.5, -0.5,  0.5, 1.,
                              -0.5,  0.5,  0.5, 1.,
                              -0.5,  0.5,  0.5, 1.,
                              0.5, -0.5,  0.5, 1.,
                              0.5,  0.5,  0.5, 1.,

                              0.5, -0.5, -0.5, 1.,
                              -0.5, -0.5, -0.5, 1.,
                              0.5,  0.5, -0.5, 1.,
                              0.5,  0.5, -0.5, 1.,
                              -0.5, -0.5, -0.5, 1.,
                              -0.5,  0.5, -0.5, 1. ],
                            dtype = numpy.float32 )

box_normals = numpy.array( [  0., -1., 0., 0., -1., 0., 0., -1., 0.,
                              0., -1., 0., 0., -1., 0., 0., -1., 0.,

                              0., 1., 0., 0., 1., 0., 0., 1., 0.,
                              0., 1., 0., 0., 1., 0., 0., 1., 0.,

                              -1., 0., 0., -1., 0., 0., -1., 0., 0.,
                              -1., 0., 0., -1., 0., 0., -1., 0., 0.,

                              1., 0., 0., 1., 0., 0., 1., 0., 0.,
                              1., 0., 0., 1., 0., 0., 1., 0., 0.,

                              0., 0., 1., 0., 0., 1., 0., 0., 1.,
                              0., 0., 1., 0., 0., 1., 0., 0., 1.,

                              0., 0., -1., 0., 0., -1., 0., 0., -1.,
                              0., 0., -1., 0., 0., -1., 0., 0., -1. ],
                           dtype = numpy.float32 )


vertexVBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertexVBO)
glBufferData(GL_ARRAY_BUFFER, box_vertices, GL_STATIC_DRAW)

normalVBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, normalVBO)
glBufferData(GL_ARRAY_BUFFER, box_normals, GL_STATIC_DRAW)


VAO = glGenVertexArrays(1)
glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, vertexVBO)
glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * 4, None)   # I should be able to have 0 for penultimate arg
glEnableVertexAttribArray(0)

glBindBuffer(GL_ARRAY_BUFFER, normalVBO)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)   # (same comment)
glEnableVertexAttribArray(1)

def drawfunc():
    sys.stderr.write("Drawing\n")
    glClearColor(0., 0., 0., 0.)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    glUseProgram(progid)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 3*box_num_triangles)

    err = glGetError()
    if err != GL_NO_ERROR:
        sys.stderr.write("Error {} drawing: {}\n".format(err, gluErrorString(err)))
        sys.exit(-1)
    
    glutSwapBuffers()
    glutPostRedisplay()
    
glutDisplayFunc(drawfunc)

glutMainLoop()
