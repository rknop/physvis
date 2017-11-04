#/usr/bin/python3
# -*- coding: utf-8 -*-

# Futzing based on openglbook.com

import sys
import numpy
import math

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


# ======================================================================

class Subject:
    class __init__(self, *args, *kwargs):
        super().__init__(*args, *kwargs)
        self.listeners = []

    class __del__(self):
        for listener in self.listeners:
            listener("destruct", self)

    class broadcast(self, message):
        for listener in self.listeners:
            listener.receive_message(message, self)

    class add_listener(self, listener):
        if not listener in self.listeners:
            self.listeners.append(listener)

    class remove_listener(self, listener):
        self.listeners = [x for x in self.listeners if x != listener]

class Observer:
    class __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def receive_message(self, message, subject):
        pass


# ======================================================================
        
class GLUTContext(Observer):

    # ======================================================================
    # Class methods

    @classmethod
    def class_init(cls):
        if hasattr(GLUTContext, '_already_is_initialized') and GLUTContext._already_is_initialized != None:
            return
        GLUTContext._full_init = False
        
        glutInit(len(sys.argv), sys.argv)
        glutInitContextVersion(3, 3)
        glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

        GLUTContext._already_is_initialized = True

    @classmethod
    def post_init(cls):
        if GLUTContext._full_init:
            return
        GLUTContext._full_init = True

        GLUTContext.idle_funcs = []
        glutIdleFunc(lambda: cls.idle())

        
    @classmethod
    def add_idle_func(cls, func):
        GLUTContext.idle_funcs.append(func)

    @classmethod
    def remove_idle_func(cls, func):
        GLUTContext.idle_funcs = [x for x in cls.idle_funcs if x != func]

    @classmethod
    def idle(cls):
        for func in GLUTContext.idle_funcs:
            func()
        glutPostRedisplay()

    # ======================================================================
    # Instance methods
    
    def __init__(self, width=500, height=400, title="GLUT", *args, *kwargs):
        super().__init__(*args, *kwargs)
        self.__class__.class_init()
        
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

        glutSetWindow(self.window)
        glutReshapeFunc(lambda width, height : self.resize2d(width, height))
        glutDisplayFunc(lambda : self.draw())
        glutTimerFunc(0, lambda val : self.timer(val), 0)
        glutCloseFunc(lambda : self.cleanup())

        self.create_shaders()
        # self.create_VBO()

        self.objects = []
        
        self.__class__.post_init()

    def receive_message(self, message, subject):
        sys.stderr.write("OMG!  Got message {} from subject {}, should do something!\n"
                         .format(message, subject))

    def add_object(self, obj):
        self.objects.append(obj)
        obj.add_listener(self)
        
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
        glDeleteBuffers(1, [self.colorbuffer])
        glDeleteBuffers(1, [self.vboarr])

        glBindVertexArray(0)
        glDeleteVertexArrays(1, [self.vtxarr])

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

    def destroy_shaders(self):
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
# ======================================================================
# ======================================================================

class Object(Subject):
    def __init__(self, context=None, position=None, rotation=None, scale=None,
                 *args, *kwargs):
        super().__init__(*args, *kwargs)

        self.context = context
        
        if position is None:
            self._position = numpy.array([0., 0., 0.])
        else:
            self._position = position

        if rotation is None:
            self._rotation = numpy.array([0., 0., 0., 1.])
        else:
            self._rotation = rotation

        if scale is None:
            self._scale = numpy.array([1., 1., 1.])
        else:
            self._scale = scale

        self.shader_program = None

        self.EBO = None
        self.VBO = None

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        # Verify len-3 numpy array!!
        self._position = numpy.array(value)

    @property
    def x(self):
        return self._position[0]

    @x.setter
    def x(self, value):
        self._position[0] = value

    @property
    def y(self):
        return self._position[1]

    @y.setter
    def y(self, value):
        self._position[1] = value

    @property
    def z(self):
        return self._position[2]

    @z.setter
    def z(self, value):
        self._position[2] = value

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        # Verify that it's a 4-element array!
        self._rotation = numpy.array(value)

    def set_rotation(angle = 0., ux=0., uy=0., uz=0.):
        vec = numpy.array([ ux, uy, uz ])
        vec2 = vec*vec
        if (vec2.sum() != 0.):
            vec /= math.sqrt(vec2.sum())
            self.rotation_[0:2] = vec * math.sin(angle/2.)
            self.rotation_[3] = mat.cos(angle/2.)
        else:
            self.rotation_[0:4] = numpy.array([0., 0., 0., 1.])


# ======================================================================

class Box(Object):
    def __init__(self, *args, *kwargs):
        super().__init__(*args, *kwargs)

        self.vertices = numpy.array( [-0.5, -0.5, -0.5,
                                      -0.5, -0.5,  0.5,
                                      -0.5,  0.5, -0.5,
                                      -0.5,  0.5,  0.5,
                                       0.5, -0.5, -0.5,
                                       0.5, -0.5,  0.5,
                                       0.5,  0.5, -0.5,
                                       0.5,  0.5, -0.5]
                                     dtype=numpy.float32 )
        self.indices = numpy.array( [0, 1, 2,
                                     2, 1, 3,
                                     3, 1, 5,
                                     5, 3, 7,
                                     7, 5, 4,
                                     4, 7, 6,
                                     6, 4, 0,
                                     0, 6, 2,
                                     2, 3 ,7,
                                     7, 2, 6,
                                     0, 1, 5,
                                     5, 0, 4],
                                    dtype=numpy.int32 )
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)

        self.context.add_object(self)

# ======================================================================
# ======================================================================

            
glext = GLUTContext()

sys.stderr.write("OpenGL version: {}\n".format(glGetString(GL_VERSION)))
sys.stderr.write("OpenGL renderer: {}\n".format(glGetString(GL_RENDERER)))
sys.stderr.write("OpenGL vendor: {}\n".format(glGetString(GL_VENDOR)))
sys.stderr.write("OpenGL shading language version: {}\n"
                 .format(glGetString(GL_SHADING_LANGUAGE_VERSION)))

glutMainLoop()
