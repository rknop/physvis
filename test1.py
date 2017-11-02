#/usr/bin/python3
# -*- coding: utf-8 -*-

# Futzing based on openglbook.com

import sys
import numpy

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


class GLUTContext:
    def __init__(self, width=500, height=400, title="GLUT"):
        self.width = width
        self.height = height
        self.title = title
        self.framecount = 0
        self.vtxarr = None
        self.vboarr = None
        self.colorbuffer = None
        self.vtxshdrid = None
        self.fragshdrid = None
        self.progid = None
        
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(self.title)

        # NOTE -- I think these are global so shouldn't be in the class

        glutReshapeFunc(lambda width, height : self.resize2d(width, height))
        glutDisplayFunc(lambda : self.draw())
        glutIdleFunc(lambda : self.idle())
        glutTimerFunc(0, lambda val : self.timer(val), 0)
        glutCloseFunc(lambda : self.cleanup())

        self.create_shaders()
        self.create_VBO()

    def cleanup(self):
        self.destroy_shaders()
        self.destroy_VBO()

    def create_VBO(self):
        verticies = numpy.array( [-0.8, -0.8,  0.0,  1.0,
                                   0.0,  0.8,  0.0,  1.0,
                                   0.8, -0.8,  0.0,  1.0], dtype=numpy.float32 )

        colors = numpy.array( [ 1.0, 0.0, 0.0, 1.0,
                                0.0, 1.0, 0.0, 1.0,
                                0.0, 0.0, 1.0, 1.0 ], dtype=numpy.float32 )

        err = glGetError()

        self.vtxarr = glGenVertexArrays(1)
        glBindVertexArray(self.vtxarr)

        self.vboarr = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vboarr)
        glBufferData(GL_ARRAY_BUFFER, verticies, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0);

        self.colorbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.colorbuffer)
        glBufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        err = glGetError()
        if err != GL_NO_ERROR:
            sys.stderr.write("Error {} creating VBO: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)

    def destroy_VBO(self):
        err = glGetError()
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(1, self.colorbuffer)
        glDeleteBuffers(1, self.vboarr)

        glBindVertexArray(0)
        glDeleteVertexArrayds(1, self.vtxarr)

        err = glGetError()
        if err != GL_NO_ERROR:
            sys.stderr.write("Error {} destroying VBO: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)


    def create_shaders(self):
        err = glGetError()

        vertex_shader = """
#version 330

layout(location=0) in vec4 in_Position;
layout(location=1) in vec4 in_Color;
out vec4 ex_Color;

void main(void)
{
  gl_Position = in_Position;
  ex_Color = in_Color;
}"""

        fragment_shader = """
#version 330

in vec4 ex_Color;
out vec4 out_Color;

void main(void)
{
  out_Color = ex_Color;
}"""
        
        self.vtxshdrid = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vtxshdrid, vertex_shader)
        glCompileShader(self.vtxshdrid)

        self.fragshdrid = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fragshdrid, fragment_shader)
        glCompileShader(self.fragshdrid)

        # sys.stderr.write("{}\n".format(glGetShaderInfoLog(self.vtxshdrid)))
        # sys.stderr.write("{}\n".format(glGetShaderInfoLog(self.fragshdrid)))
        
        self.progid = glCreateProgram()
        glAttachShader(self.progid, self.vtxshdrid)
        glAttachShader(self.progid, self.fragshdrid)
        glLinkProgram(self.progid)

        if glGetProgramiv(self.progid, GL_LINK_STATUS) != GL_TRUE:
            sys.stderr.write("{}\n".format(glGetProgramInfoLog(self.progid)))
            sys.exit(-1)

        glUseProgram(self.progid)

        err = glGetError()
        if err != GL_NO_ERROR:
            sys.stderr.write("Error {} creating shaders: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)

    def destroy_shaders():
        err = glGetError()

        glUseProgram(0)

        glDetachShader(self.progid, self.vtxshdrid)
        glDetachShader(self.progid, self.fragshdrid)

        glDeleteShader(self.fragshdrid)
        glDeleteShader(self.vtxshdrid)

        glDeleteProgram(self.progid)

        err = glGetError()
        if err != GL_NO_ERROR:
            sys.stderr.write("Error {} creating shaders: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)

        
    def idle(self):
        glutPostRedisplay()
        
    def timer(self, val):
        sys.stderr.write("{} Frames per Second\n".format(self.framecount/2.))
        self.framecount = 0
        glutTimerFunc(2000, lambda val : self.timer(val), 0)

    def resize2d(self, width, height):
        sys.stderr.write("In resize2d w/ size {} × {}\n".format(width, height))
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
    
    def draw(self):
        glClearColor(0., 0., 0., 0.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glDrawArrays(GL_TRIANGLES, 0, 3)

        glutSwapBuffers()
        glutPostRedisplay()

        self.framecount += 1

# ======================================================================
glutInit(len(sys.argv), sys.argv)
glutInitContextVersion(3, 3)
glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
glutInitContextProfile(GLUT_CORE_PROFILE)
glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)


glext = GLUTContext()

sys.stderr.write("OpenGL version: {}\n".format(glGetString(GL_VERSION)))
sys.stderr.write("OpenGL renderer: {}\n".format(glGetString(GL_RENDERER)))
sys.stderr.write("OpenGL vendor: {}\n".format(glGetString(GL_VENDOR)))
sys.stderr.write("OpenGL shading language version: {}\n"
                 .format(glGetString(GL_SHADING_LANGUAGE_VERSION)))

glutMainLoop()
