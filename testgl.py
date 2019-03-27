#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math
import ctypes

import numpy

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


# https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
#
# My quarternions are [ sin(θ/2)*x̂, sin(θ/2)*ŷ, sin(θ/2)*ẑ, cos(θ/2) ]
def quarternion_multiply(p, q):
    if len(p) == 3:
        px, py, pz = numpy.array(p)
        pr = 0.
    else:
        px, py, pz, pr = numpy.array(p)
    qx, qy, qz, qr = numpy.array(q)
    return numpy.array( [ pr*qx + px*qr + py*qz - pz*qy,
                          pr*qy - px*qz + py*qr + pz*qx,
                          pr*qz + px*qy - py*qx + pz*qr,
                          pr*qr - px*qx - py*qy - pz*qz ] , dtype=numpy.float32 )

def rotate(q, angle, axis):
    s = math.sin(angle/2.)
    c = math.cos(angle/2.)
    newq = numpy.array( [axis[0]*s, axis[1]*s, axis[2]*s, c] )
    return quarternion_multiply(newq, q)

# Here, theta is off of the z-axis, and phi is in the x-y plane
def get_model_matrix(x, y, z, sx, sy, sz, theta, phi):
    q = rotate(numpy.array([0., 0., 0., 1.]), theta, [0., 1., 0.])
    q = rotate(q, phi, [0., 0., 1.])
    s = 1./( (q*q).sum() )
    rot = numpy.matrix(
        [[ 1.-2*s*(q[1]*q[1]+q[2]*q[2]) ,    2*s*(q[0]*q[1]-q[2]*q[3]) ,    2*s*(q[0]*q[2]+q[1]*q[3])],
         [    2*s*(q[0]*q[1]+q[2]*q[3]) , 1.-2*s*(q[0]*q[0]+q[2]*q[2]) ,    2*s*(q[1]*q[2]-q[0]*q[3])],
         [    2*s*(q[0]*q[2]-q[1]*q[3]) ,    2*s*(q[1]*q[2]+q[0]*q[3]) , 1.-2*s*(q[0]*q[0]+q[1]*q[1])]],
        dtype=numpy.float32)
    mat = numpy.matrix( [[ sx, 0., 0., 0. ],
                         [ 0., sy, 0., 0. ],
                         [ 0., 0., sz, 0. ],
                         [ 0., 0., 0., 1.]], dtype=numpy.float32 )
    rotation = numpy.identity(4, dtype=numpy.float32)
    rotation[0:3, 0:3] = rot.T
    mat *= rotation
    translation = numpy.identity(4, dtype=numpy.float32)
    translation[3, 0:3] = numpy.array([x, y, z])
    mat *= translation
    invmat3 = numpy.linalg.inv(mat[0:3, 0:3]).T
    return (mat, invmat3)
    
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
uniform mat4 model_unif;
uniform mat3 model_normal_unif;

layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Normal;
layout(location=2) in mat4 model;
layout(location=6) in mat3 model_normal;
out vec3 aNormal;

void main(void)
{
  gl_Position =  projection * view * model * in_Position;
  aNormal = model_normal * in_Normal;
  // gl_Position = projection * view * in_Position;
  // aNormal = in_Normal;
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


# Rotate and move the thing

(mat, invmat3) = get_model_matrix(0.5, 0., 0., 1., 1., 1., 0., 0.)
loc = glGetUniformLocation(progid, "model_unif")
glUniformMatrix4fv(loc, 1, GL_FALSE, mat)
loc = glGetUniformLocation(progid, "model_normal_unif")
glUniformMatrix3fv(loc, 1, GL_FALSE, invmat3)

sys.stderr.write("\nmat:\n{}\n".format(mat))
sys.stderr.write("\nivmat3:\n{}\n".format(invmat3))

matbox = numpy.empty( (box_num_triangles, 4, 4), dtype=numpy.float32)
invmat3box = numpy.empty( (box_num_triangles, 3, 3), dtype=numpy.float32 )
for i in range(box_num_triangles):
    matbox[i, :, :] = mat
    invmat3box[i, :, :] = invmat3

vertexVBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertexVBO)
glBufferData(GL_ARRAY_BUFFER, box_vertices, GL_STATIC_DRAW)

normalVBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, normalVBO)
glBufferData(GL_ARRAY_BUFFER, box_normals, GL_STATIC_DRAW)

matVBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, matVBO)
glBufferData(GL_ARRAY_BUFFER, matbox, GL_STATIC_DRAW)

normmatVBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, normmatVBO)
glBufferData(GL_ARRAY_BUFFER, invmat3box, GL_STATIC_DRAW)

VAO = glGenVertexArrays(1)
glBindVertexArray(VAO)

glBindBuffer(GL_ARRAY_BUFFER, vertexVBO)
glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(0)

glBindBuffer(GL_ARRAY_BUFFER, normalVBO)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
glEnableVertexAttribArray(1)

glBindBuffer(GL_ARRAY_BUFFER, matVBO)
glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4*4*4, ctypes.c_void_p(0))
glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4*4*4, ctypes.c_void_p(4*4*1))
glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4*4*4, ctypes.c_void_p(4*4*2))
glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4*4*4, ctypes.c_void_p(4*4*3))
glEnableVertexAttribArray(2)
glEnableVertexAttribArray(3)
glEnableVertexAttribArray(4)
glEnableVertexAttribArray(5)
glVertexAttribDivisor(2, 1)        # Need this to keep the thing from jumping over four matrices
glVertexAttribDivisor(3, 1)        #  instead of just one for each vertex
glVertexAttribDivisor(4, 1)
glVertexAttribDivisor(5, 1)

err = glGetError()
if err != GL_NO_ERROR:
    sys.stderr.write("Error {} binding model array: {}\n".format(err, gluErrorString(err)))
    sys.exit(-1)

glBindBuffer(GL_ARRAY_BUFFER, normmatVBO)
glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 4*3*3, ctypes.c_void_p(0))
glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, 4*3*3, ctypes.c_void_p(4*3*1))
glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, 4*3*3, ctypes.c_void_p(4*3*2))
glEnableVertexAttribArray(6)
glEnableVertexAttribArray(7)
glEnableVertexAttribArray(8)
glVertexAttribDivisor(6, 1)
glVertexAttribDivisor(7, 1)
glVertexAttribDivisor(8, 1)

err = glGetError()
if err != GL_NO_ERROR:
    sys.stderr.write("Error {} binding model normal array: {}\n".format(err, gluErrorString(err)))
    sys.exit(-1)

def drawfunc():
    # sys.stderr.write("Drawing\n")
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
