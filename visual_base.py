#/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math
import time
import queue
import threading

import numpy
import numpy.linalg

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

time_of_last_rate_call = None
def rate(fps):
    global time_of_last_rate_call
    if time_of_last_rate_call is None:
        time.sleep(1./fps)
    else:
        sleeptime = time_of_last_rate_call + 1./fps - time.perf_counter()
        if sleeptime > 0:
            time.sleep(sleeptime)

    time_of_last_rate_call = time.perf_counter()
        
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
            
    
def gl_version_info():
    sys.stderr.write("OpenGL version: {}\n".format(glGetString(GL_VERSION)))
    sys.stderr.write("OpenGL renderer: {}\n".format(glGetString(GL_RENDERER)))
    sys.stderr.write("OpenGL vendor: {}\n".format(glGetString(GL_VENDOR)))
    sys.stderr.write("OpenGL shading language version: {}\n"
                     .format(glGetString(GL_SHADING_LANGUAGE_VERSION)))

# ======================================================================

class color(object):
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

class Subject(object):
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

class Observer(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def receive_message(self, message, subject):
        pass


# ======================================================================
        
class GLUTContext(Observer):

    # ======================================================================
    # Class methods

    @staticmethod
    def class_init(object):
        if hasattr(GLUTContext, '_already_is_initialized') and GLUTContext._already_is_initialized is not None:
            return
        GLUTContext._post_init = False
        GLUTContext._full_init = False
        
        if not hasattr(GLUTContext, '_default_context') or GLUTContext._default_context is None:
            GLUTContext._default_context = object

        GLUTContext.threadlock = threading.Lock()
        
        glutInit(len(sys.argv), sys.argv)
        glutInitContextVersion(3, 3)
        glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
        glutInitContextProfile(GLUT_CORE_PROFILE)
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

        GLUTContext._already_is_initialized = True

    # I'm afraid of a race condition on _post_init
    @staticmethod
    def post_init():
        if GLUTContext._post_init:
            while not GLUTContext._full_init:
                time.sleep(0.1)
            return

        GLUTContext._post_init = True
        GLUTContext._full_init = False

        GLUTContext.idle_funcs = []
        GLUTContext.things_to_run = queue.Queue()

        glutIdleFunc(lambda: GLUTContext.idle())

        # sys.stderr.write("Starting GLUT thread...\n")
        GLUTContext.thread = threading.Thread(target = lambda : GLUTContext.thread_main() )
        GLUTContext.thread.start()
        while not GLUTContext._full_init:
            time.sleep(0.1)
        
    # There's a race condition here on idle_funcs and things_to_run
    @staticmethod
    def thread_main():
        glutMainLoop()
        
    @staticmethod
    def add_idle_func(func):
        GLUTContext.idle_funcs.append(func)

    @staticmethod
    def remove_idle_func(func):
        GLUTContext.idle_funcs = [x for x in GLUTContext.idle_funcs if x != func]

    @staticmethod
    def run_glcode(func):
        if not hasattr(GLUTContext, "_post_init") or not GLUTContext._post_init:
            func()
        else:
            while not GLUTContext._full_init:
                time.sleep(0.1)
            GLUTContext.things_to_run.put(func)
            
    @staticmethod
    def idle():
        try:
            while not GLUTContext.things_to_run.empty():
                func = GLUTContext.things_to_run.get()
                func()
        except queue.Empty:
            pass
        
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
        GLUTContext.class_init(self)
        
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

        self.objects = []

        GLUTContext.run_glcode(lambda : self.gl_init())
        
    def gl_init(self):
        glutInitWindowSize(self.width, self.height)
        glutInitWindowPosition(0, 0)
        self.window = glutCreateWindow(self.title)
        glutSetWindow(self.window)
        glutVisibilityFunc(lambda state : self.window_visibility_handler(state))

        GLUTContext.post_init()

        GLUTContext.run_glcode(lambda : self.gl_init2())
        
    def gl_init2(self):
        self.create_shaders()
        glutReshapeFunc(lambda width, height : self.resize2d(width, height))
        glutDisplayFunc(lambda : self.draw())
        glutTimerFunc(0, lambda val : self.timer(val), 0)
        glutCloseFunc(lambda : self.cleanup())

    def window_visibility_handler(self, state):
        if state != GLUT_VISIBLE:
            return
        glutSetWindow(self.window)
        GLUTContext.threadlock.acquire()
        GLUTContext._full_init = True
        GLUTContext.threadlock.release()
        glutVisibilityFunc(None)
        
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
        # sys.stderr.write("Perspective matrix:\n{}\n".format(matrix))
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
                    glBindVertexArray(obj.VAO)
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.EBO)
                    glDrawElements(GL_TRIANGLES, obj.num_triangles*3, GL_UNSIGNED_SHORT, None)
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

        if context is None:
            if not hasattr(GLUTContext, "_default_context") or GLUTContext._default_context is None:
                GLUTContext._default_context = GLUTContext()
            self.context = GLUTContext._default_context
        else:
            self.context = context

        self.is_elements = False         # True if using EBO

        self._rotation = numpy.array( [0., 0., 0., 1.] )    # Identity quarternion
        
        if position is None:
            self._position = numpy.array([0., 0., 0.])
        else:
            self._position = numpy.array(position)

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
        
        if axis is not None:
            self.axis = numpy.array(axis)

        if up is not None:
            self.up = numpy.array(up)
            
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, position must have 3 elements.")
            sys.exit(20)
        self._position = numpy.array(value)
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
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, scale must have 3 elements.")
            sys.exit(20)
        self._scale = numpy.array(value)
        self.update_model_matrix()

    @property
    def sx(self):
        return self._scale[0]

    @sx.setter
    def sx(self, value):
        self._scale[0] = value
        self.update_model_matrix()
        
    @property
    def sy(self):
        return self._scale[1]

    @sy.setter
    def sy(self, value):
        self._scale[1] = value
        self.update_model_matrix()
        
    @property
    def sz(self):
        return self._scale[2]

    @sz.setter
    def sz(self, value):
        self._scale[2] = value
        self.update_model_matrix()
        
    @property
    def axis(self):
        v = numpy.array([1., 0., 0.], dtype=numpy.float32)
        q = self._rotation
        qinv = q.copy()
        qinv[0:3] *= -1.
        qinv /= (q*q).sum()
        return quarternion_multiply(q, quarternion_multiply(v, qinv))[0:3]

    @axis.setter
    def axis(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, axis must have 3 elements.")
            sys.exit(20)
        R = math.sqrt(value[0]*value[0] + value[2]*value[2])
        theta = math.atan2(value[1], R)
        phi = math.atan2(value[2], value[0])
        q1 = numpy.array([ 0., 0., numpy.sin(theta/2.), numpy.cos(theta/2.)])
        q2 = numpy.array([ 0., numpy.sin(phi/2.), 0., numpy.cos(phi/2.)])
        self._rotation = quarternion_multiply(q2, q1)
        self.update_model_matrix()

    @property
    def up(self):
        # return self._up
        sys.stderr.write("up not implemented\n")
        return None

    @up.setter
    def up(self, value):
        # if len(value) != 3:
        #     sys.stderr.write("ERROR, up must have 3 elements.")
        # self._up = numpy.array(up)
        # self.update_model_matrix()
        sys.stderr.write("up not implemented\n")

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if len(value) != 4:
            sys.sderr.write("rotation is a quarternion, eneds 4 values\n")
            sys.exit(20)
        self._rotation = numpy.array(value)
        self.update_model_matrix()
        
    def rotate(self, angle, axis=None, origin=None):
        if origin is not None:
            sys.stderr.write("WARNING: Rotations not around object origin aren't currently supported.\n")
        if axis is None:
            axis = self.axis
        s = math.sin(angle/2.)
        c = math.cos(angle/2.)
        q = numpy.array( [axis[0]*s, axis[1]*s, axis[2]*s, c] )
        self.rotation = quarternion_multiply(q, self.rotation)
        
    def update_model_matrix(self):
        q = self._rotation
        s = 1./( (q*q).sum() )
        rot = numpy.matrix(
            [[ 1.-2*s*(q[1]*q[1]+q[2]*q[2]) ,    2*s*(q[0]*q[1]-q[2]*q[3]) ,    2*s*(q[0]*q[2]+q[1]*q[3])],
             [    2*s*(q[0]*q[1]+q[2]*q[3]) , 1.-2*s*(q[0]*q[0]+q[2]*q[2]) ,    2*s*(q[1]*q[2]-q[0]*q[3])],
             [    2*s*(q[0]*q[2]-q[1]*q[3]) ,    2*s*(q[1]*q[2]+q[0]*q[3]) , 1.-2*s*(q[0]*q[0]+q[1]*q[1])]],
            dtype=numpy.float32)
        mat = numpy.matrix( [[ self._scale[0], 0., 0., 0. ],
                             [ 0., self._scale[1], 0., 0. ],
                             [ 0., 0., self._scale[2], 0. ],
                             [ 0., 0., 0., 1.]], dtype=numpy.float32 )
        rotation = numpy.identity(4)
        rotation[0:3, 0:3] = rot.T
        mat *= rotation
        translation = numpy.identity(4)
        translation[3, 0:3] = self._position
        mat *= translation
        self.model_matrix = mat
        # sys.stderr.write("model matrix: {}\n".format(self.model_matrix))
        # sys.stderr.write("scale: {}\n".format(self._scale))

            
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
        if len(rgb) != 3:
            sys.stderr.write("ERROR!  Need all of r, g, and b for color.\n")
            sys.exit(20)
        self._color[0:3] = numpy.array(rgb)
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
    def __init__(self, length=1., width=1., height=1., *args, **kwargs):
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

        self.scale = numpy.array( [length, height, width] ) 
        
        self.context.add_object(self)

    @property
    def length(self):
        return self.sx

    @length.setter
    def length(self, value):
        self.sx = value

    @property
    def width(self):
        return self.sz

    @width.setter
    def width(self, value):
        self.sz = value

    @property
    def height(self):
        return self.sy

    @height.setter
    def height(self, value):
        self.sy = value
        
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

class Icosahedron(Object):
    def __init__(self, radius=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertices = numpy.zeros( 4*12, dtype=numpy.float32 )
        self.edges = numpy.zeros( (30, 2), dtype=numpy.uint16 )
        self.faces = numpy.zeros( (20, 3), dtype=numpy.uint16 )

        # Vertices: 1 at top (+x), 5 next row, 5 next row, 1 at bottom

        r = 0.5
        self._radius = r
        self.vertices[0:4] = [r, 0., 0., 1.]
        angles = numpy.arange(0, 2*math.pi, 2*math.pi/5)
        for i in range(len(angles)):
            self.vertices[4+4*i:8+4*i] = [ 0.447213595499958*r,
                                           0.8944271909999162*r*math.cos(angles[i]),
                                           0.8944271909999162*r*math.sin(angles[i]),
                                           1.]
            self.vertices[24+4*i:28+4*i] = [-0.447213595499958*r,
                                             0.8944271909999162*r*math.cos(angles[i]+angles[1]/2.),
                                             0.8944271909999162*r*math.sin(angles[i]+angles[1]/2.),
                                            1.]
        self.vertices[44:48] = [-r, 0., 0., 1.]

        self.edges[0:5, :]   = [ [0, 1], [0, 2], [0, 3], [0, 4], [0, 5] ]
        self.edges[5:10, :]  = [ [1, 2], [2, 3], [3, 4], [4, 5], [5, 1] ]
        self.edges[10:20, :] = [ [1, 6], [2, 6], [2, 7], [3, 7], [3, 8],
                                 [4, 8], [4, 9], [5, 9], [5, 10], [1, 10] ]
        self.edges[20:25, :] = [ [6, 7], [7, 8], [8, 9], [9, 10], [10, 6] ]
        self.edges[25:30, :] = [ [6, 11], [7, 11], [8, 11], [9, 11], [10, 11] ]

        self.faces[0:5, :] = [ [0, 5, 1], [1, 6, 2], [2, 7, 3], [3, 8, 4], [4, 9, 0] ]
        self.faces[5:10, :] = [ [5, 10, 11], [6, 12, 13], [7, 14, 15], [8, 16, 17],
                                [9, 18, 19] ]
        self.faces[10:15, :] = [ [20, 12, 11], [21, 14, 13], [22, 16, 15],
                                 [23, 18, 17], [24, 10, 19] ]
        self.faces[15:20, :] = [ [25, 26, 20], [26, 27, 21], [27, 28, 22],
                                 [28, 29, 23], [29, 25, 24] ]

        self.subdivide()
        self.subdivide()

        self.normals = numpy.zeros( 3*len(self.vertices)//4, dtype=numpy.float32 )
        for i in range(len(self.vertices)//4):
            self.normals[3*i:3*i+3] = ( self.vertices[4*i:4*i+3] /
                                        math.sqrt( (self.vertices[4*i:4*i+3]**2).sum() ))

        self.indices = numpy.zeros( self.faces.shape[0] * 3, dtype=numpy.uint16 )
        v = numpy.zeros(6, dtype=numpy.uint16)
        for i in range(self.faces.shape[0]):
            dex = 0
            for j in range(3):
                for k in range(2):
                    v[dex] = self.edges[self.faces[i, j], k]
                    dex += 1
            if len(numpy.unique(v)) != 3:
                sys.stderr.write("ERROR with face {}, {} vertices: {}\n"
                                 .format(i, len(numpy.unique(v)), numpy.unique(v)))
                sys.exit(20)
            if ( ( self.edges[self.faces[i, 0], 0] == self.edges[self.faces[i, 1], 0] ) or
                 ( self.edges[self.faces[i, 0], 0] == self.edges[self.faces[i, 1], 1] ) ):
                self.indices[3*i+0] = self.edges[self.faces[i, 0], 1]
                self.indices[3*i+1] = self.edges[self.faces[i, 0], 0]
            else:
                self.indices[3*i+0] = self.edges[self.faces[i, 0], 0]
                self.indices[3*i+1] = self.edges[self.faces[i, 0], 1]
            if ( ( self.edges[self.faces[i, 1], 0] == self.edges[self.faces[i, 0], 0] ) or
                 ( self.edges[self.faces[i, 1], 0] == self.edges[self.faces[i, 0], 1] ) ):
                self.indices[3*i+2] = self.edges[self.faces[i, 1], 1]
            else:
                self.indices[3*i+2] = self.edges[self.faces[i, 1], 0]

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

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices, GL_STATIC_DRAW)

        sys.stderr.write("{} triangles, {} indices, {} vertices\n"
                         .format(self.faces.shape[0], len(self.indices),
                                 len(self.vertices)//4))
        
        self.num_triangles = self.faces.shape[0]
        self.is_elements = True

        self.context.add_object(self)

    def subdivide(self):
        newverts = numpy.zeros( len(self.vertices) + 4*self.edges.shape[0], dtype=numpy.float32 )
        newverts[0:len(self.vertices)] = self.vertices
        numoldverts = len(self.vertices) // 4
        
        for i in range(self.edges.shape[0]):
            vertex = 0.5 * ( self.vertices[ 4*self.edges[i, 0] : 4*self.edges[i, 0]+4 ] +
                             self.vertices[ 4*self.edges[i, 1] : 4*self.edges[i, 1]+4 ] )
            vertex[0:3] *= self._radius / math.sqrt( (vertex[0:3]**2).sum() )
            newverts[len(self.vertices) + 4*i : len(self.vertices) + 4*i + 4] = vertex

        newedges = numpy.zeros( (2*self.edges.shape[0] + 3*self.faces.shape[0], 2 ) ,
                                dtype=numpy.uint16 )
        newfaces = numpy.zeros( (4*self.faces.shape[0], 3) , dtype=numpy.uint16 )

        for en in range(self.edges.shape[0]):
            newedges[2*en, 0] = self.edges[en, 0]
            newedges[2*en, 1] = numoldverts+en
            newedges[2*en+1, 0] = self.edges[en, 1]
            newedges[2*en+1, 1] = numoldverts+en
        for fn in range(self.faces.shape[0]):
            newedges[2*self.edges.shape[0] + 3*fn + 0, 0] = numoldverts + self.faces[fn, 0]
            newedges[2*self.edges.shape[0] + 3*fn + 0, 1] = numoldverts + self.faces[fn, 1]
            newedges[2*self.edges.shape[0] + 3*fn + 1, 0] = numoldverts + self.faces[fn, 1]
            newedges[2*self.edges.shape[0] + 3*fn + 1, 1] = numoldverts + self.faces[fn, 2]
            newedges[2*self.edges.shape[0] + 3*fn + 2, 0] = numoldverts + self.faces[fn, 2]
            newedges[2*self.edges.shape[0] + 3*fn + 2, 1] = numoldverts + self.faces[fn, 0]

        for fn in range(self.faces.shape[0]):
            if ( self.edges[self.faces[fn, 0], 0] == self.edges[self.faces[fn, 1], 0] or
                 self.edges[self.faces[fn, 0], 0] == self.edges[self.faces[fn, 1], 1] ):
                corner1 = self.edges[self.faces[fn, 0], 1]
                corner2 = self.edges[self.faces[fn, 0], 0]
            else:
                corner1 = self.edges[self.faces[fn, 0], 0]
                corner2 = self.edges[self.faces[fn, 0], 1]
            if ( self.edges[self.faces[fn, 1], 0] == self.edges[self.faces[fn, 0], 0] or
                 self.edges[self.faces[fn, 1], 0] == self.edges[self.faces[fn, 0], 1] ):
                corner3 = self.edges[self.faces[fn, 1], 1]
            else:
                corner3 = self.edges[self.faces[fn, 1], 0]

            if newedges[2*self.faces[fn, 0], 0] == corner1:
                edge1l = 2*self.faces[fn, 0]
                edge1r = 2*self.faces[fn, 0] + 1
            else:
                edge1l = 2*self.faces[fn, 0] + 1
                edge1r = 2*self.faces[fn, 0]
            if newedges[2*self.faces[fn, 1], 0] == corner2:
                edge2l = 2*self.faces[fn, 1]
                edge2r = 2*self.faces[fn, 1] + 1
            else:
                edge2l = 2*self.faces[fn, 1] + 1
                edge2r = 2*self.faces[fn, 1]
            if newedges[2*self.faces[fn, 2], 0] == corner3:
                edge3l = 2*self.faces[fn, 2]
                edge3r = 2*self.faces[fn, 2] + 1
            else:
                edge3l = 2*self.faces[fn, 2] + 1
                edge3r = 2*self.faces[fn, 2]
            mid1 = 2*self.edges.shape[0] + 3*fn
            mid2 = 2*self.edges.shape[0] + 3*fn + 1
            mid3 = 2*self.edges.shape[0] + 3*fn + 2
                
            newfaces[4*fn,     :] = [edge1l, mid3, edge3r]
            newfaces[4*fn + 1, :] = [edge1r, edge2l, mid1]
            newfaces[4*fn + 2, :] = [mid2, edge2r, edge3l]
            newfaces[4*fn + 3, :] = [mid1, mid2, mid3]

        self.vertices = newverts
        self.edges = newedges
        self.faces = newfaces
                    
# ======================================================================

def main():
    sys.stderr.write("Making box.\n")
    box = Box(position = (0., 0., 0.), axis = (1., -1., 1.), color=color.red,
              length = 1.5, width=0.25, height=0.25)
    sys.stderr.write("Making Icosahedron.\n")
    d20 = Icosahedron(position= (2., 0., 0.), color=color.green)
    theta = math.pi/4.
    phi = 0.
    fps = 30
    dphi = 2*math.pi/(4.*fps)

    gl_version_info()

    while True:
        phi += dphi
        if phi > 2.*math.pi:
            phi -= 2.*math.pi
        box.axis = numpy.array( [math.sin(theta)*math.cos(phi),
                                 math.sin(theta)*math.sin(phi),
                                 math.cos(theta)] )
        d20.x = 2.*math.cos(phi)
        if math.sin(phi)>0.:
            d20.rotate(dphi)
        else:
            d20.rotate(-dphi)
        rate(fps)
        


    
    
# ======================================================================

if __name__ == "__main__":
    main()

