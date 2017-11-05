#/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import numpy
import numpy.linalg
import math
import time

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


# ======================================================================

class color:
    red = numpy.array( [1., 0., 0.] )
    green = numpy.array( [0., 1., 0.] )
    blue = numpy.array( [0., 0., 1.] )
    yellow = numpy.array( [1., 1., 0.] )
    cyan = numpy.array( [0., 1., 1.] )
    magenta = numpy.array( [1., 0., 1.] )
    orange = numpy.array( [1., 0., 0.5] )
    black = numpy.array( [0., 0. ,0.] )
    white = numpy.array( [1., 1., 1.] )

# ======================================================================

class Subject:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.listeners = []

    def __del__(self):
        for listener in self.listeners:
            listener("destruct", self)

    def broadcast(self, message):
        for listener in self.listeners:
            listener.receive_message(message, self)

    def add_listener(self, listener):
        if not listener in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners = [x for x in self.listeners if x != listener]

class Observer:
    def __init__(self, *args, **kwargs):
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

    @staticmethod
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
        
    # ======================================================================
    # Instance methods
    
    def __init__(self, width=500, height=400, title="GLUT", *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        self.fov = math.pi/4.
        self.clipnear = 0.1
        self.clipfar = 100.
        
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(self.title)

        glutSetWindow(self.window)
        glutReshapeFunc(lambda width, height : self.resize2d(width, height))
        glutDisplayFunc(lambda : self.draw())
        glutTimerFunc(0, lambda val : self.timer(val), 0)
        glutCloseFunc(lambda : self.cleanup())

        self.create_shaders()

        self.objects = []
        
        self.__class__.post_init()

    def receive_message(self, message, subject):
        sys.stderr.write("OMG!  Got message {} from subject {}, should do something!\n"
                         .format(message, subject))

    def add_object(self, obj):
        sys.stderr.write("Adding object {}.\n".format(obj))
        self.objects.append(obj)
        obj.add_listener(self)

    def remove_object(self, obj):
        sys.stderr.write("Removing object {}.\n".format(obj))
        self.objects = [x for x in self.objects if x != obj]
        obj.remove_listener(self)
        
    def cleanup(self):
        self.destroy_shaders()
        # DO MORE
        
    def create_shaders(self):
        err = glGetError()

        vertex_shader = """
#version 330

uniform mat4 model;
uniform mat3 model_normal;
uniform mat4 view;
uniform mat4 projection;

layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Normal;
out vec3 aNormal;

void main(void)
{
  gl_Position =  projection * view * model * in_Position;
  aNormal = model_normal * in_Normal;
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
        
        self.vtxshdrid = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vtxshdrid, vertex_shader)
        glCompileShader(self.vtxshdrid)

        self.fragshdrid = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fragshdrid, fragment_shader)
        glCompileShader(self.fragshdrid)

        sys.stderr.write("{}\n".format(glGetShaderInfoLog(self.vtxshdrid)))
        sys.stderr.write("{}\n".format(glGetShaderInfoLog(self.fragshdrid)))
        
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

        loc = glGetUniformLocation(self.progid, "ambientcolor")
        glUniform3fv(loc, 1, numpy.array([0.2, 0.2, 0.2]))
        loc = glGetUniformLocation(self.progid, "light1color")
        glUniform3fv(loc, 1, numpy.array([0.8, 0.8, 0.8]))
        loc = glGetUniformLocation(self.progid, "light1dir")
        glUniform3fv(loc, 1, numpy.array([0.22, 0.44, 0.88]))
        loc = glGetUniformLocation(self.progid, "light2color")
        glUniform3fv(loc, 1, numpy.array([0.3, 0.3, 0.3]))
        loc = glGetUniformLocation(self.progid, "light2dir")
        glUniform3fv(loc, 1, numpy.array([-0.88, -0.22, -0.44]))

        self.set_perspective(self.fov, self.width/self.height, self.clipnear, self.clipfar)
        self.set_camera_position(0., 0., 5.)

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
            sys.stderr.write("Error {} destroying shaders: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)

    def set_perspective(self, fov, aspect, near, far):
        matrix = self.perspective_matrix(fov, aspect, near,far)
        sys.stderr.write("Perspective matrix:\n{}\n".format(matrix))
        glUseProgram(self.progid)
        projection_location = glGetUniformLocation(self.progid, "projection")
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, matrix)
        
    def set_camera_position(self, x, y, z):
        matrix = numpy.matrix([[ 1., 0., 0., 0.],
                               [ 0., 1., 0., 0.],
                               [ 0., 0., 1., 0.],
                               [-x, -y, -z,  1.]] ,dtype=numpy.float32)
        glUseProgram(self.progid)
        view_location = glGetUniformLocation(self.progid, "view")
        glUniformMatrix4fv(view_location, 1, GL_FALSE, matrix)

    def set_model_matrix(self, matrix):
        glUseProgram(self.progid)
        model_location = glGetUniformLocation(self.progid, "model")
        glUniformMatrix4fv(model_location, 1, GL_FALSE, matrix)
        modnorm_location = glGetUniformLocation(self.progid, "model_normal")
        glUniformMatrix3fv(modnorm_location, 1, GL_FALSE, numpy.linalg.inv(matrix[0:3, 0:3]).T)
        
    def timer(self, val):
        sys.stderr.write("{} Frames per Second\n".format(self.framecount/2.))
        self.framecount = 0
        glutTimerFunc(2000, lambda val : self.timer(val), 0)

    def resize2d(self, width, height):
        sys.stderr.write("In resize2d w/ size {} × {}\n".format(width, height))
        self.width = width
        self.height = height
        glViewport(0, 0, self.width, self.height)
        self.set_perspective(self.fov, self.width/self.height, self.clipnear, self.clipfar)
    
    def draw(self):
        glClearColor(0., 0., 0., 0.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        
        # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glUseProgram(self.progid)
        
        for obj in self.objects:
            if obj.visible:
                # sys.stderr.write("Trying to draw {} in color {}\n".format(obj, obj._color))
                self.set_model_matrix(obj.model_matrix)
                color_location = glGetUniformLocation(self.progid, "color")
                glUniform4fv(color_location, 1, obj._color)
                if obj.is_elements:
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.EBO)
                    glDrawElements(GL_TRIANGLES, len(obj.indices), GL_UNSIGNED_INT, None)
                else:
                    glBindVertexArray(obj.VAO)
                    glDrawArrays(GL_TRIANGLES, 0, obj.num_triangles*3)

        # err = glGetError()
        # if err != GL_NO_ERROR:
        #     sys.stderr.write("Error {} drawing: {}\n".format(err, gluErrorString(err)))
        #     sys.exit(-1)

        glutSwapBuffers()
        glutPostRedisplay()

        self.framecount += 1

# ======================================================================
# ======================================================================
# ======================================================================

class Object(Subject):
    def __init__(self, context=None, position=None, axis=None, up=None, scale=None,
                 color=None, opacity=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.context = context

        self.is_elements = False         # True if using EBO
        
        if position is None:
            self._position = numpy.array([0., 0., 0.])
        else:
            self._position = numpy.array(position)

        if axis is None:
            self._axis = numpy.array([1., 0., 0.])
        else:
            self._axis = numpy.array(axis)

        if up is None:
            self._up = numpy.array([0., 1., 0.])
        else:
            self._up = numpy.array(up)
            
        if scale is None:
            self._scale = numpy.array([1., 1., 1.])
        else:
            self._scale = scale

        self.shader_program = None

        if color is None and opacity is None:
            self._color = numpy.array( [1., 1., 1., 1.], dtype=numpy.float32 )
        elif color is None:
            self._color = numpy.array( [1., 1., 1., opacity], dtype=numpy.float32 )
        else:
            self._color = numpy.empty(4)
            self._color[0:3] = numpy.array(color)
            if opacity is None:
                self._color[3] = 1.
            else:
                self._color[3] = opacity
                
        
        self.EBO = None
        self.VBO = None
        self.VAO = None
        self.num_triangles = 0
        self._visible = True

        self.model_matrix = numpy.array( [ [ 1., 0., 0., 0. ],
                                           [ 0., 1., 0., 0. ],
                                           [ 0., 0., 1., 0. ],
                                           [ 0., 0., 0., 1. ] ], dtype=numpy.float32)
        self.update_model_matrix()
        
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, position must have 3 elements.")
            sys.exit(20)
        self.position = numpy.array(value)
        self.update_model_matrix()
        
    @property
    def x(self):
        return self._position[0]

    @x.setter
    def x(self, value):
        self._position[0] = value
        self.update_model_matrix()

    @property
    def y(self):
        return self._position[1]

    @y.setter
    def y(self, value):
        self._position[1] = value
        self.update_model_matrix()

    @property
    def z(self):
        return self._position[2]

    @z.setter
    def z(self, value):
        self._position[2] = value
        self.update_model_matrix()

    @property
    def axis(self):
        return self._axis

    @axis.setter
    def axis(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, axis must have 3 elements.")
            sys.exit(20)
        self._axis = numpy.array(value)
        self.update_model_matrix()

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, up must have 3 elements.")
        self._up = numpy.array(up)
        self.update_model_matrix()

    def rotate_to(self, theta, phi):
        self._axis = numpy.array( [math.sin(theta)*math.cos(phi),
                                   math.sin(theta)*math.sin(phi),
                                   math.cos(theta)] )
        self.update_model_matrix()
        
    def update_model_matrix(self):
        horiz = math.sqrt(self._axis[0]**2 + self._axis[2]**2)
        theta1 = math.atan2(self._axis[1], horiz)
        theta2 = math.atan2(self._axis[2], self._axis[0])
        rot = numpy.matrix( [[ math.cos(theta1),  math.sin(theta1), 0.],
                             [-math.sin(theta1),  math.cos(theta1), 0.],
                             [        0.       ,          0.      , 1.]], dtype=numpy.float32)
        rot *= numpy.matrix( [[ math.cos(theta2), 0.,  math.sin(theta2)],
                              [       0.       ,  1.,        0.        ],
                              [-math.sin(theta2), 0.,  math.cos(theta2)]], dtype=numpy.float32)
        self._up /= math.sqrt( (self._up**2).sum() )
        # ROB IMPLEMENT UP!
        self.model_matrix[0:3, 0:3] = rot.T
        self.model_matrix[3, 0:3] = self._position
        self.model_matrix[0:3, 3] = 0.
        self.model_matrix[3, 3] = 1.
        # sys.stderr.write("model matrix:\n{}\n".format(self.model_matrix))

            
    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        sys.stderr.write("In visible setter\n")
        value = bool(value)
        if value == self._visible: return

        self._visible = value
        if value == True:
            self.context.add_object(self)
        else:
            self.context.remove_object(self)

    @property
    def color(self):
        return self._color[0:3]

    @color.setter
    def color(self, rgb):
        self._color[0:3] = rgb
        self.color_update()

    @property
    def opacity(self):
        return self.color[3]

    @opacity.setter
    def opacity(self, alpha):
        self._color[3] = alpha
        self.color_update()

    def color_update(self):
        glUseProgram(self.context.progid)
        color_location = glGetUniformLocation(self.context.progid, "color")
        glUniform4fv(color_location, 1, self.color)

        # sys.stderr.write("{}\n".format(glGetProgramInfoLog(self.context.progid)))
        # err = glGetError()
        # sys.stderr.write("After color_update, err={} ({})\n".format(err, gluErrorString(err)))
        

    def __del__(self):
        self.visible = False
        self.destroy()

    def destroy(self):
        pass

# ======================================================================

class Box(Object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertices = numpy.array( [ -0.5, -0.5,  0.5, 1.,
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
                                        0.5,  0.5, -0.5, 1.,
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

        self.normals = numpy.array( [ 0., -1., 0., 0., -1., 0., 0., -1., 0.,
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

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)
                          
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        self.normalbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalbuffer)
        glBufferData(GL_ARRAY_BUFFER, self.normals, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        self.num_triangles = 12
        
        self.context.add_object(self)

    def destroy(self):
        sys.stderr.write("Destroying box {}\n".format(self))
        glBindVertexArray(self.VAO)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(0)
        glDeleteBuffers(1, [self.VBO])
        glDeleteBuffers(1, [self.normalbuffer])
        glDeleteVertexArrays(1, [self.VAO])

        err = glGetError()
        if err != GL_NO_ERROR:
            sys.stderr.write("Error {} destroying a Box: {}\n"
                             .format(err, gluSerrorString(err)))
            
            

# ======================================================================
# ======================================================================

class ThingDoer:
    def __init__(self, glext):
        self.glext = glext
        self.box = None
        self.theta = math.pi/4.
        self.phi = 0.
        self.dphi = 2*math.pi/120.
        self.dt = 1./30.
        
    def dothings(self):
        # sys.stderr.write("time.perf_counter() = {}\n".format(time.perf_counter()))

        if self.box is None:
            # self.box = Box(self.glext)
            self.box = Box(self.glext, position = (0., 0., 0.), axis = (1., -1., 1.), color=color.red)
            self.boxcreatetime = time.perf_counter()
            self.nextchange = self.boxcreatetime + self.dt
            self.green = True


        if time.perf_counter() > self.nextchange:
            self.phi += self.dphi
            if self.phi > 2.*math.pi: self.phi -= 2.*math.pi
            self.box.rotate_to(self.theta, self.phi)
            self.nextchange += self.dt
        

# ======================================================================
        
glext = GLUTContext()

sys.stderr.write("OpenGL version: {}\n".format(glGetString(GL_VERSION)))
sys.stderr.write("OpenGL renderer: {}\n".format(glGetString(GL_RENDERER)))
sys.stderr.write("OpenGL vendor: {}\n".format(glGetString(GL_VENDOR)))
sys.stderr.write("OpenGL shading language version: {}\n"
                 .format(glGetString(GL_SHADING_LANGUAGE_VERSION)))

thingdoer = ThingDoer(glext)
glext.add_idle_func( lambda: thingdoer.dothings() )

glutMainLoop()
