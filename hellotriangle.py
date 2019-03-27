#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Primary source: https://learnopengl.com/Getting-started/Hello-Triangle
#
# But GLUT from elewhere
#  (maybe: http://www.lighthouse3d.com/tutorials/glut-tutorial/)

import sys
import math

import numpy

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


# ========================================

glutInit(len(sys.argv), sys.argv)
glutInitContextVersion(3, 3)
glutInitWindowPosition(-1, -1)
glutInitWindowSize(600, 600)
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

window = glutCreateWindow("Hello Triangle")


# ========================================

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}
"""

vertexShader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertexShader, vertex_shader_source)
glCompileShader(vertexShader)

if glGetShaderiv(vertexShader, GL_COMPILE_STATUS) != GL_TRUE:
    sys.stderr.write("Vertex shader compilation error: {}\n"
                     .format(glGetShaderInfoLog(vertexShader)))
    sys.exit(-1)
    

fragment_shader_source = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}
"""

fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragmentShader, fragment_shader_source)
glCompileShader(fragmentShader)

if glGetShaderiv(fragmentShader, GL_COMPILE_STATUS) != GL_TRUE:
    sys.stderr.write("Fragment shader compilation error: {}\n"
                     .format(glGetShaderInfoLog(fragmentShader)))
    sys.exit(-1)
    
shaderProgram = glCreateProgram()
glAttachShader(shaderProgram, vertexShader)
glAttachShader(shaderProgram, fragmentShader)
glLinkProgram(shaderProgram)

if glGetProgramiv(shaderProgram, GL_LINK_STATUS) != GL_TRUE:
    sys.stderr.write("Shader link error: {}\n"
                     .format(glGetProgramInfoLog(shaderProgram)))
    sys.exit(-1)


glUseProgram(shaderProgram)


VAO = glGenVertexArrays(1)
glBindVertexArray(VAO)

vertices = numpy.array( [ -0.5, -0.5, 0.,
                          0.5, -0.5, 0.,
                          0., 0.5, 0. ], dtype=numpy.float32 );

VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None);
glEnableVertexAttribArray(0)

# ========================================

def renderScene():
    glClearColor(0., 0., 0., 0.)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram)
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, 3)

    glutSwapBuffers()

# ========================================

glutDisplayFunc(renderScene)

glutMainLoop()

    
