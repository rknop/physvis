#/usr/bin/python3
# -*- coding: utf-8 -*-

# NOTE!  I just rudely GLEnable(GL_EXT_separate_shader_objects) below
#  without testing that it's there.  I should do better.
# (Actually, I think that's commented out now.)

import sys
import math
import time
import queue
import threading
import random
import uuid
import ctypes

import numpy
import numpy.linalg

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
### from OpenGL.GL.ARB.separate_shader_objects import *

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
# My quaternions are [ sin(θ/2)*ux, sin(θ/2)*uy, sin(θ/2)*uz, cos(θ/2) ]
#  Representing a rotation of θ about a unit vector (ux, uy,u z)
#
# To rotate a vector v by quaterion q, do qvq¯¹, where q¯¹ can be
#  simply composed by flipping the sign of the first 3 elements of
#  q and dividing by q·q (see quaternion_rotate)
#
# If p and q are both quaternions, then their product
#  represents rotation q followed by rotation p
#
# All quaternions must be normalized, or there will be unexpected results.
#
# NOTE!  You MUST pass p and q as numpy arrays, or things might be sad

def quaternion_multiply(p, q):
    if len(p) == 3:
        px, py, pz = p
        pr = 0.
    else:
        px, py, pz, pr = p
    qx, qy, qz, qr = q
    return numpy.array( [ pr*qx + px*qr + py*qz - pz*qy,
                          pr*qy - px*qz + py*qr + pz*qx,
                          pr*qz + px*qy - py*qx + pz*qr,
                          pr*qr - px*qx - py*qy - pz*qz ] , dtype=numpy.float32 )

# Rotate vector p by quaternion q
def quaternion_rotate(p, q):
    qinv = q.copy()
    qinv[0:3] *= -1.
    return quaternion_multiply(q, quaternion_multiply(p, qinv))[0:3]

# ======================================================================

class color(object):
    red = numpy.array( [1., 0., 0.] )
    green = numpy.array( [0., 1., 0.] )
    blue = numpy.array( [0., 0., 1.] )
    yellow = numpy.array( [1., 1., 0.] )
    cyan = numpy.array( [0., 1., 1.] )
    magenta = numpy.array( [1., 0., 1.] )
    orange = numpy.array( [1., 0.5, 0.] )
    black = numpy.array( [0., 0. ,0.] )
    white = numpy.array( [1., 1., 1.] )

# ======================================================================

class Subject(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._id = uuid.uuid4()
        self.listeners = []
        # ROB!  Print warnings about unknown arguments

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
        super().__init__()
        # ROB!  Print errors about unknown arguments

    def receive_message(self, message, subject):
        pass


# ======================================================================
#
# One object collection encapsulates a set of objects that can
#  all be drawn with the same shader.
#
# Shaders take as input for each vertex of each triangle
#  location  (4 floats per vertex)
#  normal    (3 floats per vertex)
#  index     (1 index per vertex)
#
# The object collection points to a set of 3 VBOs with this information
# for each vertex of each object.  There's a single VBO so that the
# whole damn thing can be drawn in one call to OpenGL for efficiency
# purposes.  This means that I've got to do all sorts of memory
# management manually in order to keep track of which data goes with
# which object.  (I could reduce the amount of data per object by using
# EBOs, but that would also make the data management more complicated.)
#
# Shaders also have a few arrays of uniforms, one element of the array
# for each object; the input "index" is points into this arary.
#
#  model matrix  (16 floats per object)
#  model normal matrix (something like an inverse)  (12* floats per object)
#  color     (4 floats per vertex)
#
#  * really, it's a mat3, so you'd think 9 floats per object.  However,
#  The OpenGL std140 layout means that things are aligned on vec4
#  bounadries, so there's an extra "junk" float at the end of each
#  row of the matrix.
#
# I'm gonna have to remind myself why location and color need 4, not 3,
# floats.  (For location, it's so that the transformation matrices can
# be 4×4 to allow offsets as well as rotations and scales.)  (For color,
# alpha?  It's not implemented, but maybe that's what I was thinking.)
#

class GLObjectCollection(Observer):

    def __init__(self, context, shader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxnumobjs = 512       # Must match the array length in the shader!!!!  Rob, do better.
        self.maxnumtris = 32768

        self.curnumtris = 0      # These four must be kept consistent.
        self.objects = []
        self.object_index = []
        self.object_triangle_index = []

        self.draw_as_lines = False
        self.shader = shader
        self.context = context

        self.is_initialized = False
        context.run_glcode(lambda : self.initglstuff())

        while not self.is_initialized:
            time.sleep(0.1)

    def initglstuff(self):
        self.vertexbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        # 4 bytes per float * 4 floats per vertex * 3 vertices per triangle
        glBufferData(GL_ARRAY_BUFFER, 4 * 4 * 3 * self.maxnumtris, None, GL_STATIC_DRAW)

        self.normalbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normalbuffer)
        # 4 bytes per float * 3 floats per vertex * 3 vertices per triangle
        glBufferData(GL_ARRAY_BUFFER, 4 * 3 * 3 * self.maxnumtris, None, GL_STATIC_DRAW)

        self.objindexbuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.objindexbuffer)
        # 4 bytes per int * 1 int per vertex * 3 vertices per triangle
        glBufferData(GL_ARRAY_BUFFER, 4 * 1 * 3 * self.maxnumtris, None, GL_STATIC_DRAW)
        
        self.modelmatrixbuffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
        # 4 bytes per float * 16 floats per object
        glBufferData(GL_UNIFORM_BUFFER, 4 * 16 * self.maxnumobjs, None, GL_DYNAMIC_DRAW)

        self.modelnormalmatrixbuffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
        # 4 bytes per float * 9 floats per object
        #  BUT!  Because of std140 layout, there's actually 12 floats per object,
        #    as the alignment of each row of the matrix is like a vec4 rather than a vec3
        glBufferData(GL_UNIFORM_BUFFER, 4 * 12 * self.maxnumobjs, None, GL_DYNAMIC_DRAW)

        self.colorbuffer = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, self.colorbuffer)
        # 4 bytes per float * 4 floats per object
        glBufferData(GL_UNIFORM_BUFFER, 4 * 4 * self.maxnumobjs, None, GL_DYNAMIC_DRAW)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self.normalbuffer)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, self.objindexbuffer)
        glVertexAttribIPointer(2, 1, GL_INT, 0, None)
        glEnableVertexAttribArray(2)

        dex = glGetUniformBlockIndex(self.shader.progid, "ModelMatrix")
        sys.stderr.write("ModelMatrix block index (progid={}): {}\n".format(dex, self.shader.progid))
        glUniformBlockBinding(self.shader.progid, dex, 0);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.modelmatrixbuffer)

        dex = glGetUniformBlockIndex(self.shader.progid, "ModelNormalMatrix")
        sys.stderr.write("ModelNormalMatrix block index: {}\n".format(dex))
        glUniformBlockBinding(self.shader.progid, dex, 1);
        glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.modelnormalmatrixbuffer)

        dex = glGetUniformBlockIndex(self.shader.progid, "Colors")
        sys.stderr.write("Colors block index: {}\n".format(dex))
        glUniformBlockBinding(self.shader.progid, dex, 2);
        glBindBufferBase(GL_UNIFORM_BUFFER, 2, self.colorbuffer)
        
        # In the past, I was passing a model matrix for each
        # and every vertex.  That was profligate.  I'm leaving this
        # comment here, though, as it's got a pointer to docs how I made that work.
        # See https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_instanced_arrays.txt
        # and http://sol.gfxile.net/instancing.html

        self.is_initialized = True

    def add_object(self, obj):
        # Make sure not to double-add
        for cur in self.objects:
            if cur._id == obj._id:
                return

        if len(self.objects) >= self.maxnumobjs:
            raise Exception("Error, I can currently only handle {} objects.".format(self.maxnumobjs))
        if self.curnumtris + obj.num_triangles > self.maxnumtris:
            raise Exception("Error, I can currently only handle {} triangles.".format(self.maxnumtris))

        self.object_triangle_index.append(self.curnumtris)
        self.object_index.append(len(self.objects))
        self.objects.append(obj)
        obj.add_listener(self)
        self.curnumtris += obj.num_triangles
        sys.stderr.write("Up to {} objects, {} triangles.\n".format(len(self.objects), self.curnumtris))

        # I will admit to not fully understanding how lambdas work
        # I originally had lambda : self.push_all_object_info(len(self.objects)-1); however
        # the argument didn't seem to be evaluated at the time of the lambda creation,
        # but rather later.  Calculating n first seemed to fix the issue.

        n = len(self.objects) - 1 
        self.context.run_glcode(lambda : self.push_all_object_info(n))

    def update_object_matrix(self, obj):
        found = False
        # sys.stderr.write("Going to try to update object matrix for {}\n".format(obj._id))
        for i in range(len(self.objects)):
            if self.objects[i]._id == obj._id:
                found = True
                break

        if not found:
            # sys.stderr.write("...not found\n")
            return

        # sys.stderr.write("...found at {}!\n".format(i))
        # sys.stderr.write("\nmatrixdata:\n{}\n".format(obj.matrixdata))
        # sys.stderr.write("\nnormalmatrixdata:\n{}\n".format(obj.normalmatrixdata))

        self.context.run_glcode(lambda : self.do_update_object_matrix(i, obj))

    def do_update_object_matrix(self, dex, obj):
        with GLUTContext._threadlock:
            glBindBuffer(GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
            glBufferSubData(GL_UNIFORM_BUFFER, self.object_index[dex]*4*16, obj.model_matrix.flatten())
            glBindBuffer(GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
            glBufferSubData(GL_UNIFORM_BUFFER, self.object_index[dex]*4*12, obj.inverse_model_matrix.flatten())
            glutPostRedisplay()
        

    # Updates positions of verticies and directions of normals.
    # Can NOT change the number of vertices
    def update_object_vertices(self, obj):
        found = False
        # sys.stderr.write("Going to try to update object vertex data for {}\n".format(obj._id))
        for i in range(len(self.objects)):
            if self.objects[i]._id == obj._id:
                found = True
                break

        if not found:
            # sys.stderr.write("...not found\n")
            return

        self.context.run_glcode(lambda : self.do_update_object_vertex(i, obj))

    def do_update_object_vertex(self, dex, obj):
        with GLUTContext._threadlock:
            glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
            glBufferSubData(GL_ARRAY_BUFFER, self.object_triangle_index[dex]*4*4*3, obj.vertexdata)
            glBindBuffer(GL_ARRAY_BUFFER, self.normalbuffer)
            glBufferSubData(GL_ARRAY_BUFFER, self.object_triangle_index[dex]*4*3*3, obj.normaldata)
            glutPostRedisplay()
            
    def push_all_object_info(self, dex):

        # sys.stderr.write("Pushing object info for index {} (with {} triangles, at offset {}).\n"
        #                  .format(dex, self.objects[dex].num_triangles,
        #                          self.object_triangle_index[dex]))
        # sys.stderr.write("\nvertexdata: {}\n".format(self.objects[dex].vertexdata))
        # sys.stderr.write("\nnormaldata: {}\n".format(self.objects[dex].normaldata))
        # sys.stderr.write("\ncolordata: {}\n".format(self.objects[dex].colordata))
        # sys.stderr.write("\nmatrixdata: {}\n".format(self.objects[dex].matrixdata))
        # sys.stderr.write("\nnormalmatrixdata: {}\n".format(self.objects[dex].normalmatrixdata))
        # sys.exit(20)

        # sys.stderr.write("Pushing vertexdata for obj {}\n".format(dex))
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        glBufferSubData(GL_ARRAY_BUFFER, self.object_triangle_index[dex]*4*4*3, self.objects[dex].vertexdata)

        # sys.stderr.write("Pushing normaldata for obj {}\n".format(dex))
        glBindBuffer(GL_ARRAY_BUFFER, self.normalbuffer)
        glBufferSubData(GL_ARRAY_BUFFER, self.object_triangle_index[dex]*4*3*3, self.objects[dex].normaldata)

        objindexcopies = numpy.empty(self.objects[dex].num_triangles*3, dtype=numpy.int32)
        objindexcopies[:] = self.object_index[dex]
        # sys.stderr.write("Pushing object_index for obj {}\n".format(dex))
        # sys.stderr.write("objindexcopies = {}\n".format(objindexcopies))
        glBindBuffer(GL_ARRAY_BUFFER, self.objindexbuffer)
        glBufferSubData(GL_ARRAY_BUFFER, self.object_triangle_index[dex]*4*1*3, objindexcopies)
        
        # sys.stderr.write("Pushing model_matrix for obj {} which has index {} and byte position {}\n"
        #                  .format(dex, self.object_index[dex], self.object_index[dex]*4*16))
        glBindBuffer(GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
        glBufferSubData(GL_UNIFORM_BUFFER, self.object_index[dex]*4*16, self.objects[dex].model_matrix.flatten())

        # sys.stderr.write("Pushing inverse_model_Matrix for obj {}\n".format(dex))
        glBindBuffer(GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
        glBufferSubData(GL_UNIFORM_BUFFER, self.object_index[dex]*4*12, self.objects[dex].inverse_model_matrix.flatten())

        # ROB!  Make this not use the _ variable
        # sys.stderr.write("Pushing _color for obj {}\n".format(dex))
        glBindBuffer(GL_UNIFORM_BUFFER, self.colorbuffer)
        glBufferSubData(GL_UNIFORM_BUFFER, self.object_index[dex]*4*4, self.objects[dex]._color)

        glutPostRedisplay()


    # Never call this directly!  It should only be called from within the
    #   draw method of a GLUTContext
    def draw(self):
        with GLUTContext._threadlock:
            glUseProgram(self.shader.progid)
            self.shader.set_perspective(self.context._fov, self.context.width/self.context.height,
                                        self.context._clipnear, self.context._clipfar)
            self.shader.set_camera_posrot(self.context._camx, self.context._camy, self.context._camz,
                                          self.context._camtheta, self.context._camphi)

            if self.draw_as_lines:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBindVertexArray(self.VAO)
            # sys.stderr.write("About to draw {} triangles\n".format(self.curnumtris))
            glDrawArrays(GL_TRIANGLES, 0, self.curnumtris*3)

    # Need to add ability to update color
    def receive_message(self, message, subject):
        # sys.stderr.write("Got message \"{}\" from {}\n".format(message, subject._id))
        if message == "update matrix":
            self.update_object_matrix(subject)
        if message == "update vertices":
            self.update_object_vertices(subject)

# ======================================================================
#
# AK.  It's been a while, and I didn't comment.
#
# I think I was trying to set this up so taht you couold have more
# than one GLUTContext object... but then the class stuff seems
# to be setting things up so there's only one GLUT Context, and I'm
# not 100% clear on what a GLUT Context fully means.
#
# I added passing "instance" to class_init_2, as it seemed that
# I had to initialize GLUT in the same therad as the mainloop.  I suspect
# that the whole setup of the thing is now more complicated than it
# needs to be, and it certainly won't work with more than one window
# now.  Rethinking is needed.

class GLUTContext(Observer):
    _threadlock = threading.RLock()
    _class_init_1 = False
    _class_init_2 = False

    _instance = None       # Is GLUTContext a singleton?  Geez, dunno.
    
    # ======================================================================
    # Class methods

    @staticmethod
    def class_init(object):
        sys.stderr.write("Starting class_init\n")

        with GLUTContext._threadlock:
            if GLUTContext._class_init_1:
                return

            if not hasattr(GLUTContext, '_default_context') or GLUTContext._default_context is None:
                GLUTContext._default_context = object

            glutInit(len(sys.argv), sys.argv)
            glutInitContextVersion(3, 3)
            glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
            glutInitContextProfile(GLUT_CORE_PROFILE)
            glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS)
            glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

            ### res = glInitSeparateShaderObjectsARB() # ROB check error return
            ### glEnable(res)

            GLUTContext._class_init_1 = True

    @staticmethod
    def class_init_2(instance):
        sys.stderr.write("Starting class_init_2\n")
        GLUTContext._instance = instance
        with GLUTContext._threadlock:
            if not GLUTContext._class_init_1:
                raise Exception("class_init_2() called with _class_init_1 False")

            if GLUTContext._class_init_2:
                return

            GLUTContext.idle_funcs = []
            GLUTContext.things_to_run = queue.Queue()

            sys.stderr.write("Starting GLUT thread...\n")
            GLUTContext.thread = threading.Thread(target = lambda : GLUTContext.thread_main(instance) )
            GLUTContext.thread.start()

            GLUTContext._class_init_2 = True

    # There's a race condition here on idle_funcs and things_to_run
    @staticmethod
    def thread_main(instance):
        sys.stderr.write("Starting thread_main\n")
        glutInitWindowSize(instance.width, instance.height)
        glutInitWindowPosition(0, 0)
        instance.window = glutCreateWindow(instance.title)
        glutSetWindow(instance.window)
        glutIdleFunc(lambda: GLUTContext.idle())
        glutMainLoop()

    @staticmethod
    def add_idle_func(func):
        GLUTContext.idle_funcs.append(func)

    @staticmethod
    def remove_idle_func(func):
        GLUTContext.idle_funcs = [x for x in GLUTContext.idle_funcs if x != func]

    @staticmethod
    def run_glcode(func):
        # sys.stderr.write("Starting run_glcode\n")
        GLUTContext.things_to_run.put(func)

    @staticmethod
    def idle():
        with GLUTContext._threadlock:
            try:
                while not GLUTContext.things_to_run.empty():
                    func = GLUTContext.things_to_run.get()
                    func()
            except queue.Empty:
                pass

            for func in GLUTContext.idle_funcs:
                func()

    # It seems to be unhappy if you call this outside
    #  of a proper OpenGL Context.  Instead call
    #  the gl_version_info method of a GLUTContext instance.
    @staticmethod
    def do_gl_version_info():
        sys.stderr.write("OpenGL version: {}\n".format(glGetString(GL_VERSION)))
        sys.stderr.write("OpenGL renderer: {}\n".format(glGetString(GL_RENDERER)))
        sys.stderr.write("OpenGL vendor: {}\n".format(glGetString(GL_VENDOR)))
        sys.stderr.write("OpenGL shading language version: {}\n"
                         .format(glGetString(GL_SHADING_LANGUAGE_VERSION)))

    # ======================================================================
    # Instance methods

    def __init__(self, width=500, height=400, title="GLUT", *args, **kwargs):
        super().__init__(*args, **kwargs)
        sys.stderr.write("Starting __init__")
        GLUTContext.class_init(self)

        self.window_is_initialized = False
        self.width = width
        self.height = height
        self.title = title
        self.framecount = 0
        self.vtxarr = None
        self.vboarr = None
        self.colorbuffer = None

        self._camx = 0.
        self._camy = 0.
        self._camz = 5.
        self._camtheta = math.pi/2.
        self._camphi = 0.

        self._fov = math.pi/4.
        self._clipnear = 0.1
        self._clipfar = 1000.

        self._mousex0 = 0.
        self._mousey0 = 0.
        self._origtheta = 0.
        self._origphi = 0.
        self._origcamz = 0.

        self.object_collections = []

        GLUTContext.class_init_2(self)

        GLUTContext.run_glcode(lambda : self.gl_init())

        while not self.window_is_initialized:
            time.sleep(0.1)

        self.object_collections.append(GLObjectCollection(self, Shader.get("Basic Shader", self)))

        sys.stderr.write("Exiting __init__\n")

    def gl_init(self):
        sys.stderr.write("Starting gl_init\n")
        glutSetWindow(self.window)
        glutMouseFunc(lambda button, state, x, y : self.mouse_button_handler(button, state, x, y))
        glutReshapeFunc(lambda width, height : self.resize2d(width, height))
        glutDisplayFunc(lambda : self.draw())
        glutVisibilityFunc(lambda state : self.window_visibility_handler(state))
        # Right now, the timer just prints FPS
        # glutTimerFunc(0, lambda val : self.timer(val), 0)
        glutCloseFunc(lambda : self.cleanup())
        self.window_is_initialized = True
        sys.stderr.write("Exiting gl_init\n")

    def gl_version_info(self):
        GLUTContext.run_glcode(lambda : GLUTContext.do_gl_version_info())

    def window_visibility_handler(self, state):
        if state != GLUT_VISIBLE:
            return
        glutSetWindow(self.window)
        with GLUTContext._threadlock:
            GLUTContext._full_init = True
        glutVisibilityFunc(None)

    def mouse_button_handler(self, button, state, x, y):
        if button == GLUT_RIGHT_BUTTON:
            glutSetWindow(self.window)

            if state == GLUT_UP:
                glutMotionFunc(None)
                if self._camtheta > math.pi:
                    self._camtheta = math.pi
                if self._camtheta < 0.:
                    self._camtheta = 0.
                if self._camphi > 2.*math.pi:
                    self._camphi -= 2.*math.pi
                if self._camphi < 0.:
                    self._camphi += 2.*math.pi

            elif state == GLUT_DOWN:
                self._mousex0 = x
                self._mousey0 = y
                self._origtheta = self._camtheta
                self._origphi = self._camphi
                glutMotionFunc(lambda x, y : self.rmb_moved(x, y))

        if button == GLUT_MIDDLE_BUTTON:
            glutSetWindow(self.window)

            if state == GLUT_UP:
                sys.stderr.write("MMB up\n")
                glutMotionFunc(None)

            elif state ==GLUT_DOWN:
                sys.stderr.write("MMB down\n")
                self._mousex0 = x
                self._mousey0 = y
                self._origcamz = self._camz
                glutMotionFunc(lambda x, y : self.mmb_moved(x, y))


        if (state == GLUT_UP) and ( button == 3 or button == 4):   # wheel up/down
            glutSetWindow(self.window)

            if button == 3:
                self._camz *= 0.9
            else:
                self._camz *= 1.1
            self.update_cam_posrot_gl()


    def rmb_moved(self, x, y):
        dx = x - self._mousex0
        dy = y - self._mousey0
        self._camtheta = self._origtheta - dy * math.pi / self.height
        self._camphi = self._origphi + dx * 2.*math.pi / self.width
        self.update_cam_posrot_gl()


    def mmb_moved(self, x, y):
        dy = y - self._mousey0
        self._camz = self._origcamz * 10.**(dy/self.width)
        self.update_cam_posrot_gl()


    def receive_message(self, message, subject):
        sys.stderr.write("OMG!  Got message {} from subject {}, should do something!\n"
                         .format(message, subject))

    def cleanup(self):
        sys.stderr.write("ROB!  You should actually write the cleanup method!!!!")
        pass
        # DO THINGS!!!!!!!!!!!!!!!!

    def timer(self, val):
        sys.stderr.write("{} Frames per Second\n".format(self.framecount/2.))
        self.framecount = 0
        glutTimerFunc(2000, lambda val : self.timer(val), 0)

    def resize2d(self, width, height):
        # sys.stderr.write("In resize2d w/ size {} × {}\n".format(width, height))
        self.width = width
        self.height = height
        GLUTContext.run_glcode(lambda : self.resize2d_gl())

    def resize2d_gl(self):
        glViewport(0, 0, self.width, self.height)
        for collection in self.object_collections:
            collection.shader.set_perspective(self._fov, self.width/self.height,
                                              self._clipnear, self._clipfar)

    def update_cam_posrot_gl(self):
        # sys.stderr.write("Moving camera to [{:.2f}, {:.2f}, {:.2f}], setting rotation to [{:.3f}, {:.3f}]\n"
        #                  .format(self._camx, self._camy, self._camz, self._camtheta, self._camphi))
        for collection in self.object_collections:
            collection.shader.set_camera_posrot(self._camx, self._camy, self._camz, self._camtheta, self._camphi)

    def draw(self):
        glClearColor(0., 0., 0., 0.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        for collection in self.object_collections:
            collection.draw()

            err = glGetError()
            if err != GL_NO_ERROR:
                sys.stderr.write("Error {} drawing: {}\n".format(err, gluErrorString(err)))
                sys.exit(-1)

        glutSwapBuffers()
        glutPostRedisplay()

        self.framecount += 1

    def add_object(self, obj):
        # Try to figure out which collection this goes into for real
        self.object_collections[0].add_object(obj)

    def remove_object(self, obj):
        raise Exception("GAH!  Removing objects isn't implemented.")

# ======================================================================
# ======================================================================
# ======================================================================

class Shader(object):
    _basic_shader = {}

    @staticmethod
    def get(name, context):
        if name == "Basic Shader":
            with GLUTContext._threadlock:
                if ( (not context in Shader._basic_shader) or
                     (Shader._basic_shader[context] == None) ):
                    sys.stderr.write("Asking for a BasicShader\n")
                    Shader._basic_shader[context] = BasicShader(context)
            return Shader._basic_shader[context]

        else:
            raise Exception("Unknown shader \"{}\"".format(name))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progid = None

    def get_shader(self):
        return self.progid

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



class BasicShader(Shader):
    def __init__(self, context, *args, **kwargs):
        sys.stderr.write("Initializing a Basic Shader...\n")
        super().__init__(*args, **kwargs)
        self.context = context
        self._name = "Basic Shader"
        self._shaders_destroyed = False

        self.vtxshdrid = None
        self.fragshdrid = None
        self.progid = None

        GLUTContext.run_glcode(lambda : self.create_shaders())

    def create_shaders(self):
        err = glGetError()

        vertex_shader = """
#version 330

uniform mat4 view;
uniform mat4 projection;

layout (std140) uniform ModelMatrix
{
   mat4 model_matrix[512];
};

layout (std140) uniform ModelNormalMatrix
{
   mat3 model_normal_matrix[512];
};

layout (std140) uniform Colors
{
   vec4 color[512];
};

layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Normal;
layout(location=2) in int in_Index;
out vec3 aNormal;
out vec4 aColor;

void main(void)
{
  gl_Position =  projection * view * model_matrix[in_Index] * in_Position;
  aNormal = model_normal_matrix[in_Index] * in_Normal;
  aColor = color[in_Index];
}"""

        fragment_shader = """
#version 330

uniform vec3 ambientcolor;
uniform vec3 light1color;
uniform vec3 light1dir;
uniform vec3 light2color;
uniform vec3 light2dir;

in vec3 aNormal;
in vec4 aColor;
out vec4 out_Color;

void main(void)
{
  vec3 norm = normalize(aNormal);
  vec3 diff1 = max(dot(norm, light1dir), 0.) * light1color;
  vec3 diff2 = max(dot(norm, light2dir), 0.) * light2color;
  vec3 col = (ambientcolor + diff1 + diff2) * vec3(aColor);
  out_Color = vec4(col, aColor[3]);
}"""

        sys.stderr.write("\nAbout to compile shaders....\n")

        self.vtxshdrid = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(self.vtxshdrid, vertex_shader)
        glCompileShader(self.vtxshdrid)

        sys.stderr.write("{}\n".format(glGetShaderInfoLog(self.vtxshdrid)))

        self.fragshdrid = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(self.fragshdrid, fragment_shader)
        glCompileShader(self.fragshdrid)

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

        self.set_perspective(self.context._fov, self.context.width/self.context.height,
                             self.context._clipnear, self.context._clipfar)
        self.set_camera_posrot(self.context._camx, self.context._camy, self.context._camz,
                               self.context._camtheta, self.context._camphi)

    # This makes me feel very queasy.  A wait for another thread in
    #   a __del__ is probably just asking for circular references
    #   to trip you up.  *But*, I gotta run all my GL code in
    #   a single thread.  So... hurm.
    def __del__(self):
        sys.stderr.write("BasicShader __del__\n")
        GLUTContext.run_glcode(lambda : self.destroy_shaders())
        while not self._shaders_destroyed:
            time.sleep(0.1)
        sys.stderr.write("...BasicShader __del__ completed\n")

    def destroy_shaders(self):
        sys.stderr.write("BasicShader destroy_shaders\n")
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

        self._shaders_destroyed = True

    def set_perspective(self, fov, aspect, near, far):
        matrix = self.perspective_matrix(fov, aspect, near,far)
        # sys.stderr.write("Perspective matrix:\n{}\n".format(matrix))
        glUseProgram(self.progid)
        projection_location = glGetUniformLocation(self.progid, "projection")
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, matrix)
        glutPostRedisplay()
        
    def set_camera_posrot(self, x, y, z, theta, phi):
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
        # sys.stderr.write("View matrix:\n{}\n".format(matrix.T))
        glUseProgram(self.progid)
        view_location = glGetUniformLocation(self.progid, "view")
        glUniformMatrix4fv(view_location, 1, GL_FALSE, matrix.T)
        glutPostRedisplay()

# ======================================================================
# ======================================================================
# ======================================================================

class GrObject(Subject):
    def __init__(self, context=None, position=None, axis=None, up=None, scale=None,
                 color=None, opacity=None, make_trail=False, interval=10, retain=50,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._make_trail = False
        self.num_triangles = 0
        self._visible = True

        # sys.stderr.write("Starting GrObject.__init__")

        if context is None:
            if not hasattr(GLUTContext, "_default_context") or GLUTContext._default_context is None:
                GLUTContext._default_context = GLUTContext()
            self.context = GLUTContext._default_context
        else:
            self.context = context

        self.draw_as_lines = False

        self._rotation = numpy.array( [0., 0., 0., 1.] )    # Identity quaternion

        if position is None:
            self._position = numpy.array([0., 0., 0.])
        else:
            self._position = numpy.array(position)

        if scale is None:
            self._scale = numpy.array([1., 1., 1.])
        else:
            self._scale = scale

        self.colordata = None

        if color is None and opacity is None:
            self._color = numpy.array( [1., 1., 1., 1.], dtype=numpy.float32 )
        elif color is None:
            self._color = numpy.array( [1., 1., 1., opacity], dtype=numpy.float32 )
        else:
            self._color = numpy.empty(4, dtype=numpy.float32)
            self._color[0:3] = numpy.array(color)[0:3]
            if opacity is None:
                self._color[3] = 1.
            else:
                self._color[3] = opacity
        self.update_colordata()


        self.model_matrix = numpy.array( [ [ 1., 0., 0., 0. ],
                                           [ 0., 1., 0., 0. ],
                                           [ 0., 0., 1., 0. ],
                                           [ 0., 0., 0., 1. ] ], dtype=numpy.float32)
        self.inverse_model_matrix = numpy.array( [ [ 1., 0., 0., 0.],
                                                   [ 0., 1., 0., 0.],
                                                   [ 0., 0., 1., 0.] ], dtype=numpy.float32)

        self.vertexdata = None
        self.normaldata = None
        self.matrixdata = None
        self.normalmatrixdata = None

        if axis is not None:
            self.axis = numpy.array(axis)

        if up is not None:
            self.up = numpy.array(up)

        self._interval = interval
        self._nexttrail = interval
        self._retain = retain
        self.make_trail = make_trail

    def finish_init(self):
        self.update_colordata()
        self.update_model_matrix()
        self.context.add_object(self)

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
        self.update_trail()

    @property
    def x(self):
        return self._position[0]

    @x.setter
    def x(self, value):
        self._position[0] = value
        self.update_model_matrix()
        self.update_trail()

    @property
    def y(self):
        return self._position[1]

    @y.setter
    def y(self, value):
        self._position[1] = value
        self.update_model_matrix()
        self.update_trail()

    @property
    def z(self):
        return self._position[2]

    @z.setter
    def z(self, value):
        self._position[2] = value
        self.update_model_matrix()
        self.update_trail()

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
        return quaternion_rotate([self._scale[0], 0., 0.], self._rotation)
        # v = numpy.array([self._scale[0], 0., 0.], dtype=numpy.float32)
        # qinv = q.copy()
        # qinv[0:3] *= -1.
        # qinv /= (q*q).sum()
        # return quaternion_multiply(q, quaternion_multiply(v, qinv))[0:3]

    @axis.setter
    def axis(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, axis must have 3 elements.")
            sys.exit(20)
        R = math.sqrt(value[0]*value[0] + value[2]*value[2])
        theta = math.atan2(value[1], R)
        phi = -math.atan2(value[2], value[0])
        q1 = numpy.array([ 0., 0., numpy.sin(theta/2.), numpy.cos(theta/2.)])
        q2 = numpy.array([ 0., numpy.sin(phi/2.), 0., numpy.cos(phi/2.)])
        self._rotation = quaternion_multiply(q2, q1)
        self._scale[0] = math.sqrt( value[0]*value[0] + value[1]*value[1] + value[2]*value[2] )
        self.update_model_matrix()

    @property
    def up(self):
        # return self._up
        raise Exception("up not implemented")

    @up.setter
    def up(self, value):
        # if len(value) != 3:
        #     sys.stderr.write("ERROR, up must have 3 elements.")
        # self._up = numpy.array(up)
        # self.update_model_matrix()
        raise Exception("up not implemented")

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if len(value) != 4:
            sys.sderr.write("rotation is a quaternion, needs 4 values\n")
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
        self.rotation = quaternion_multiply(q, self.rotation)

    def update_colordata(self):
        if (self.colordata is None) or (self.colordata.size != 3*4*self.num_triangles):
            self.colordata = numpy.empty(3*4*self.num_triangles, dtype=numpy.float32)
        for i in range(3*self.num_triangles):
            self.colordata[4*i:4*(i+1)] = self._color

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
        rotation = numpy.identity(4, dtype=numpy.float32)
        rotation[0:3, 0:3] = rot.T
        mat *= rotation
        translation = numpy.identity(4, dtype=numpy.float32)
        translation[3, 0:3] = self._position
        mat *= translation
        self.model_matrix[:] = mat
        self.inverse_model_matrix[0:3, 0:3] = numpy.linalg.inv(mat[0:3, 0:3]).T

        self.broadcast("update matrix")

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
            raise Exception("Setting objects not visible is broken.")
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
        self.update_colordata()

    @property
    def opacity(self):
        return self.color[3]

    @opacity.setter
    def opacity(self, alpha):
        self._color[3] = alpha
        self.update_colordata()

    @property
    def make_trail(self):
        return self._make_trail

    @make_trail.setter
    def make_trail(self, val):
        if not val:
            if self._make_trail:
                self.kill_trail()
                self._make_trail = False
        if val:
            if not self._make_trail:
                self.initialize_trail()
                self._make_trail = True

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, val):
        self._interval = val
        if self._nexttrail > self._interval:
            self._nexttrail = self._interval

    @property
    def retain(self):
        return self._retain

    @retain.setter
    def retain(self, val):
        if val != self._retain:
            self._retain = val
            if self._make_trail:
                self.initialize_trail()
                
    def kill_trail(self):
        self._trail = None

    def initialize_trail(self):
        self.kill_trail()
        sys.stderr.write("Initializing trail at position {}.\n".format(self._position))
        self._trail = Curve(color=self._color, maxpoints=self._retain, points=[ self._position ], num_edge_points=6)
        self._nexttrail = self._interval

    def update_trail(self):
        if not self._make_trail: return
        self._nexttrail -= 1
        if self._nexttrail <= 0:
            self._nexttrail = self._interval
            self._trail.add_point(self._position)

                
    def __del__(self):
        raise Exception("Rob, you really need to think about object deletion.")
        self.visible = False
        self.destroy()

    def destroy(self):
        pass

# ======================================================================

class Box(GrObject):

    @staticmethod
    def make_box_buffers(context):
        with GLUTContext._threadlock:
            if not hasattr(Box, "_box_vertices"):
                Box._box_vertices = numpy.array( [ -0.5, -0.5,  0.5, 1.,
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

                Box._box_normals = numpy.array( [  0., -1., 0., 0., -1., 0., 0., -1., 0.,
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


    def __init__(self, length=1., width=1., height=1., *args, **kwargs):
        super().__init__(*args, **kwargs)

        Box.make_box_buffers(self.context)

        self.num_triangles = 12
        self.vertexdata = Box._box_vertices
        self.normaldata = Box._box_normals

        self.length = length
        self.width = width
        self.height = height

        self.finish_init()

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
        raise Exception("OMG ROB!  You need to figure out how to destroy things!")

# ======================================================================
#
# The Icosahedron code may seem a little over-verbose.  I originally
# wrote it using indexed buffers, back when I did one GL draw call for
# each object.  (Slow.)  Now that I dump all triangles into one giant
# VAO, I don't use indexed buffers.

class Icosahedron(GrObject):

    @staticmethod
    def make_icosahedron_vertices(subdivisions=0):
        with GLUTContext._threadlock:
            if not hasattr(Icosahedron, "_vertices"):
                Icosahedron._vertices = [None, None, None, None, None]
                Icosahedron._normals = [None, None, None, None, None]
                Icosahedron._flatnormals = [None, None, None, None, None]
                Icosahedron._numvertices = [None, None, None, None, None]

            if Icosahedron._vertices[subdivisions] is None:

                sys.stderr.write("Creating icosahedron vertex data for {} subdivisions\n".format(subdivisions))

                vertices = numpy.zeros( 4*12, dtype=numpy.float32 )
                edges = numpy.zeros( (30, 2), dtype=numpy.uint16 )
                faces = numpy.zeros( (20, 3), dtype=numpy.uint16 )

                # Vertices: 1 at top (+x), 5 next row, 5 next row, 1 at bottom

                r = 1.0
                vertices[0:4] = [r, 0., 0., 1.]
                angles = numpy.arange(0, 2*math.pi, 2*math.pi/5)
                for i in range(len(angles)):
                    vertices[4+4*i:8+4*i] = [ 0.447213595499958*r,
                                              0.8944271909999162*r*math.cos(angles[i]),
                                              0.8944271909999162*r*math.sin(angles[i]),
                                              1.]
                    vertices[24+4*i:28+4*i] = [-0.447213595499958*r,
                                                0.8944271909999162*r*math.cos(angles[i]+angles[1]/2.),
                                                0.8944271909999162*r*math.sin(angles[i]+angles[1]/2.),
                                                1.]
                vertices[44:48] = [-r, 0., 0., 1.]

                edges[0:5, :]   = [ [0, 1], [0, 2], [0, 3], [0, 4], [0, 5] ]
                edges[5:10, :]  = [ [1, 2], [2, 3], [3, 4], [4, 5], [5, 1] ]
                edges[10:20, :] = [ [1, 6], [2, 6], [2, 7], [3, 7], [3, 8],
                                    [4, 8], [4, 9], [5, 9], [5, 10], [1, 10] ]
                edges[20:25, :] = [ [6, 7], [7, 8], [8, 9], [9, 10], [10, 6] ]
                edges[25:30, :] = [ [6, 11], [7, 11], [8, 11], [9, 11], [10, 11] ]

                faces[0:5, :] = [ [0, 5, 1], [1, 6, 2], [2, 7, 3], [3, 8, 4], [4, 9, 0] ]
                faces[5:10, :] = [ [5, 10, 11], [6, 12, 13], [7, 14, 15], [8, 16, 17],
                                   [9, 18, 19] ]
                faces[10:15, :] = [ [20, 12, 11], [21, 14, 13], [22, 16, 15],
                                    [23, 18, 17], [24, 10, 19] ]
                faces[15:20, :] = [ [25, 26, 20], [26, 27, 21], [27, 28, 22],
                                    [28, 29, 23], [29, 25, 24] ]

                for i in range(int(subdivisions)):
                    vertices, edges, faces = Icosahedron.subdivide(vertices, edges, faces, r)

                normals = numpy.zeros( 3*len(vertices)//4, dtype=numpy.float32 )
                for i in range(len(vertices)//4):
                    normals[3*i:3*i+3] = ( vertices[4*i:4*i+3] /
                                                math.sqrt( (vertices[4*i:4*i+3]**2).sum() ))

                flatnormals = numpy.zeros( 3 * faces.shape[0], dtype=numpy.float32 )
                for i in range(faces.shape[0]):
                    v1 = edges[faces[i, 0], 0]
                    v2 = edges[faces[i, 0], 1]
                    if ( edges[faces[i, 1], 0] == edges[faces[i, 0], 0] or
                         edges[faces[i, 1], 0] == edges[faces[i, 0], 1] ):
                        v3 = edges[faces[i, 1], 1]
                    else:
                        v3 = edges[faces[i, 1], 0]
                    x = vertices[4*v1]   + vertices[4*v2]   + vertices[4*v3]
                    y = vertices[4*v1+1] + vertices[4*v2+1] + vertices[4*v3+1]
                    z = vertices[4*v1+2] + vertices[4*v2+2] + vertices[4*v3+2]
                    vlen = math.sqrt(x*x + y*y + z*z)
                    flatnormals[3*i] = x/vlen
                    flatnormals[3*i+1] = y/vlen
                    flatnormals[3*i+2] = z/vlen
                        
                    
                rendervertices = numpy.zeros( 4*3*faces.shape[0], dtype=numpy.float32 )
                rendernormals = numpy.zeros( 3*3*faces.shape[0], dtype=numpy.float32 )
                renderflatnormals = numpy.zeros( 3*3*faces.shape[0], dtype=numpy.float32 )
                
                v = numpy.zeros(6, dtype=numpy.uint16)
                for i in range(faces.shape[0]):
                    dex = 0
                    for j in range(3):
                        renderflatnormals[ 3 * ( (i*3) + j ) + 0] = flatnormals[3*i]
                        renderflatnormals[ 3 * ( (i*3) + j ) + 1] = flatnormals[3*i+1]
                        renderflatnormals[ 3 * ( (i*3) + j ) + 2] = flatnormals[3*i+2]
                        for k in range(2):
                            v[dex] = edges[faces[i, j], k]
                            dex += 1

                    if len(numpy.unique(v)) != 3:
                        sys.stderr.write("ERROR with face {}, {} vertices: {}\n"
                                         .format(i, len(numpy.unique(v)), numpy.unique(v)))
                        sys.exit(20)

                    if ( ( edges[faces[i, 0], 0] == edges[faces[i, 1], 0] ) or
                         ( edges[faces[i, 0], 0] == edges[faces[i, 1], 1] ) ):
                        for k in range(4):
                            rendervertices[4*(3*i+0)+k] = vertices[4*edges[faces[i, 0], 1] + k]
                            rendervertices[4*(3*i+1)+k] = vertices[4*edges[faces[i, 0], 0] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+0)+k] = normals[3*edges[faces[i, 0], 1] + k]
                            rendernormals[3*(3*i+1)+k] = normals[3*edges[faces[i, 0], 0] + k]
                    else:
                        for k in range(4):
                            rendervertices[4*(3*i+0)+k] = vertices[4*edges[faces[i, 0], 0] + k]
                            rendervertices[4*(3*i+1)+k] = vertices[4*edges[faces[i, 0], 1] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+0)+k] = normals[3*edges[faces[i, 0], 0] + k]
                            rendernormals[3*(3*i+1)+k] = normals[3*edges[faces[i, 0], 1] + k]
                    if ( ( edges[faces[i, 1], 0] == edges[faces[i, 0], 0] ) or
                         ( edges[faces[i, 1], 0] == edges[faces[i, 0], 1] ) ):
                        for k in range(4):
                            rendervertices[4*(3*i+2)+k] = vertices[4*edges[faces[i, 1], 1] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+2)+k] = normals[3*edges[faces[i, 1], 1] + k]
                    else:
                        for k in range(4):
                            rendervertices[4*(3*i+2)+k] = vertices[4*edges[faces[i, 1], 0] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+2)+k] = normals[3*edges[faces[i, 1], 0] + k]

                sys.stderr.write("{} triangles, {} vertices, {} vertices in array\n"
                                 .format(faces.shape[0], len(vertices)//4, len(rendervertices)//4))

                Icosahedron._vertices[subdivisions] = rendervertices
                Icosahedron._normals[subdivisions] = rendernormals
                Icosahedron._flatnormals[subdivisions] = renderflatnormals


    @staticmethod
    def subdivide(vertices, edges, faces, r=1.0):
        newverts = numpy.zeros( len(vertices) + 4*edges.shape[0], dtype=numpy.float32 )
        newverts[0:len(vertices)] = vertices
        numoldverts = len(vertices) // 4

        for i in range(edges.shape[0]):
            vertex = 0.5 * ( vertices[ 4*edges[i, 0] : 4*edges[i, 0]+4 ] +
                             vertices[ 4*edges[i, 1] : 4*edges[i, 1]+4 ] )
            vertex[0:3] *= r / math.sqrt( (vertex[0:3]**2).sum() )
            newverts[len(vertices) + 4*i : len(vertices) + 4*i + 4] = vertex

        newedges = numpy.zeros( (2*edges.shape[0] + 3*faces.shape[0], 2 ) ,
                                dtype=numpy.uint16 )
        newfaces = numpy.zeros( (4*faces.shape[0], 3) , dtype=numpy.uint16 )

        for en in range(edges.shape[0]):
            newedges[2*en, 0] = edges[en, 0]
            newedges[2*en, 1] = numoldverts+en
            newedges[2*en+1, 0] = edges[en, 1]
            newedges[2*en+1, 1] = numoldverts+en
        for fn in range(faces.shape[0]):
            newedges[2*edges.shape[0] + 3*fn + 0, 0] = numoldverts + faces[fn, 0]
            newedges[2*edges.shape[0] + 3*fn + 0, 1] = numoldverts + faces[fn, 1]
            newedges[2*edges.shape[0] + 3*fn + 1, 0] = numoldverts + faces[fn, 1]
            newedges[2*edges.shape[0] + 3*fn + 1, 1] = numoldverts + faces[fn, 2]
            newedges[2*edges.shape[0] + 3*fn + 2, 0] = numoldverts + faces[fn, 2]
            newedges[2*edges.shape[0] + 3*fn + 2, 1] = numoldverts + faces[fn, 0]

        for fn in range(faces.shape[0]):
            if ( edges[faces[fn, 0], 0] == edges[faces[fn, 1], 0] or
                 edges[faces[fn, 0], 0] == edges[faces[fn, 1], 1] ):
                corner1 = edges[faces[fn, 0], 1]
                corner2 = edges[faces[fn, 0], 0]
            else:
                corner1 = edges[faces[fn, 0], 0]
                corner2 = edges[faces[fn, 0], 1]
            if ( edges[faces[fn, 1], 0] == edges[faces[fn, 0], 0] or
                 edges[faces[fn, 1], 0] == edges[faces[fn, 0], 1] ):
                corner3 = edges[faces[fn, 1], 1]
            else:
                corner3 = edges[faces[fn, 1], 0]

            if newedges[2*faces[fn, 0], 0] == corner1:
                edge1l = 2*faces[fn, 0]
                edge1r = 2*faces[fn, 0] + 1
            else:
                edge1l = 2*faces[fn, 0] + 1
                edge1r = 2*faces[fn, 0]
            if newedges[2*faces[fn, 1], 0] == corner2:
                edge2l = 2*faces[fn, 1]
                edge2r = 2*faces[fn, 1] + 1
            else:
                edge2l = 2*faces[fn, 1] + 1
                edge2r = 2*faces[fn, 1]
            if newedges[2*faces[fn, 2], 0] == corner3:
                edge3l = 2*faces[fn, 2]
                edge3r = 2*faces[fn, 2] + 1
            else:
                edge3l = 2*faces[fn, 2] + 1
                edge3r = 2*faces[fn, 2]
            mid1 = 2*edges.shape[0] + 3*fn
            mid2 = 2*edges.shape[0] + 3*fn + 1
            mid3 = 2*edges.shape[0] + 3*fn + 2

            newfaces[4*fn,     :] = [edge1l, mid3, edge3r]
            newfaces[4*fn + 1, :] = [edge1r, edge2l, mid1]
            newfaces[4*fn + 2, :] = [mid2, edge2r, edge3l]
            newfaces[4*fn + 3, :] = [mid1, mid2, mid3]

        return (newverts, newedges, newfaces)


    def __init__(self, radius=1., flat=False, subdivisions=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if subdivisions > 4.:
            raise Exception(">4 subdivisions is absurd. Even 4 is probably too many!!!")

        Icosahedron.make_icosahedron_vertices(subdivisions)

        self.num_triangles = len(Icosahedron._vertices[subdivisions]) // 12
        self.vertexdata = Icosahedron._vertices[subdivisions]
        if flat:
            self.normaldata = Icosahedron._flatnormals[subdivisions]
        else:
            self.normaldata = Icosahedron._normals[subdivisions]

        self.radius = radius

        self.finish_init()

        # sys.stderr.write("Created icosahedron with {} triangles.\n".format(self.num_triangles))

    @property
    def radius(self):
        return self.scale.sum()/3.

    @radius.setter
    def radius(self, r):
        self.scale = numpy.array( [r, r, r] )


class Sphere(Icosahedron):
    def __init__(self, subdivisions=2, *args, **kwargs):
        super().__init__(subdivisions=subdivisions, *args, **kwargs)


class Ellipsoid(Icosahedron):
    def __init__(self, subdivisions=2, length=1., width=1., height=1., *args, **kwargs):
        super().__init__(subdivisions=subdivisions, *args, **kwargs)

        self.length = length
        self.width = width
        self.height = height
        
    @property
    def length(self):
        return self.sx/2.

    @length.setter
    def length(self, value):
        self.sx = value/2.

    @property
    def width(self):
        return self.sz/2.

    @width.setter
    def width(self, value):
        self.sz = value/2.

    @property
    def height(self):
        return self.sy/2.

    @height.setter
    def height(self, value):
        self.sy = value/2.

# # ======================================================================

class Cylinder(GrObject):

    @staticmethod
    def make_cylinder_vertices(num_edge_points=16):
        with GLUTContext._threadlock:
            if not hasattr(Cylinder, "_vertices"):
                Cylinder._vertices = {}
                Cylinder._normals = {}

        with GLUTContext._threadlock:
            if not num_edge_points in Cylinder._vertices:
                
                # Number of triangles = 4 * num_edge_points
                #   one set of num_edge_points for each cap
                #   two sets of num_edge_points for the two triangles on the sides

                vertices = numpy.empty( 4 * 3*4*num_edge_points, dtype=numpy.float32)
                normals = numpy.empty( 3 * 3*4*num_edge_points, dtype=numpy.float32)

                # Endcaps are at ±1

                # Make an array of phis to make sure identical floats show up
                #   where they're supposed to
                phis = numpy.arange(num_edge_points+1) * 2*math.pi / num_edge_points
                phis[num_edge_points] = 0.
                dphi = phis[1] / 2.
                
                # Endcaps
                off = 3*num_edge_points
                for i in range(num_edge_points):
                    vertices[4 * (3*i+0) : 4 * (3*i+0) + 4] = [1., 0., 0., 1.]
                    vertices[4 * (3*i+1) : 4 * (3*i+1) + 4] = [1., math.cos(phis[i]), math.sin(phis[i]), 1.]
                    vertices[4 * (3*i+2) : 4 * (3*i+2) + 4] = [1., math.cos(phis[i+1]), math.sin(phis[i+1]), 1.]
                    normals[3 * (3*i+0) : 3 * (3*i+0) + 3] = [1., 0., 0.]
                    normals[3 * (3*i+1) : 3 * (3*i+1) + 3] = [1., 0., 0.]
                    normals[3 * (3*i+2) : 3 * (3*i+2) + 3] = [1., 0., 0.]
                    vertices[4 * (off+(3*i+0)) : 4 * (off+(3*i+0)) + 4] = [0., 0., 0., 1]
                    vertices[4 * (off+(3*i+1)) : 4 * (off+(3*i+1)) + 4] = [0., math.cos(phis[i]), math.sin(phis[i]), 1.]
                    vertices[4 * (off+(3*i+2)) : 4 * (off+(3*i+2)) + 4] = [0., math.cos(phis[i+1]),
                                                                           math.sin(phis[i+1]), 1.]
                    normals[3 * (off+3*i+0) : 3 * (off+3*i+0) + 3] = [-1., 0., 0.]
                    normals[3 * (off+3*i+1) : 3 * (off+3*i+1) + 3] = [-1., 0., 0.]
                    normals[3 * (off+3*i+2) : 3 * (off+3*i+2) + 3] = [-1., 0., 0.]

                # Sides
                off = 6*num_edge_points
                for i in range(num_edge_points):
                    vertices[4 * (off + 6*i + 0) : 4 * (off + 6*i + 0) + 4] = [1., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    normals[3 * (off + 6*i + 0) : 3 * (off + 6*i + 0) + 3] = [0., math.cos(phis[i]),
                                                                              math.sin(phis[i])]
                    vertices[4 * (off + 6*i + 1) : 4 * (off + 6*i + 1) + 4] = [1., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    normals[3 * (off + 6*i + 1) : 3 * (off + 6*i + 1) + 3] = [0., math.cos(phis[i+1]),
                                                                              math.sin(phis[i+1])]
                    vertices[4 * (off + 6*i + 2) : 4 * (off + 6*i + 2) + 4] = [0., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    normals[3 * (off + 6*i + 2) : 3 * (off + 6*i + 2) + 3] = [0., math.cos(phis[i+1]),
                                                                              math.sin(phis[i+1])]

                    vertices[4 * (off + 6*i + 3) : 4 * (off + 6*i + 3) + 4] = [0., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    normals[3 * (off + 6*i + 3) : 3 * (off + 6*i + 3) + 3] = [0., math.cos(phis[i+1]),
                                                                              math.sin(phis[i+1])]
                    vertices[4 * (off + 6*i + 4) : 4 * (off + 6*i + 4) + 4] = [0., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    normals[3 * (off + 6*i + 4) : 3 * (off + 6*i + 4) + 3] = [0., math.cos(phis[i]),
                                                                              math.sin(phis[i])]
                    vertices[4 * (off + 6*i + 5) : 4 * (off + 6*i + 5) + 4] = [1., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    normals[3 * (off + 6*i + 5) : 3 * (off + 6*i + 5) + 3] = [0., math.cos(phis[i]),
                                                                              math.sin(phis[i])]
                    
                Cylinder._vertices[num_edge_points] = vertices
                Cylinder._normals[num_edge_points] = normals


    def __init__(self, radius=1., num_edge_points=16, *args, **kwargs):
        super().__init__(*args, **kwargs)

        Cylinder.make_cylinder_vertices(num_edge_points)

        self.vertexdata = Cylinder._vertices[num_edge_points]
        self.normaldata = Cylinder._normals[num_edge_points]
        self.num_triangles = len(Cylinder._vertices[num_edge_points]) // 12

        self.radius = radius

        # sys.stderr.write("Made cylinder with radius {} and {} triangles.\n".format(radius, self.num_triangles))
        
        self.finish_init()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.scale = [self._scale[0], value, value]

# ======================================================================

class Cone(GrObject):

    @staticmethod
    def make_cone_vertices():
        with GLUTContext._threadlock:
            if not hasattr(Cone, "_vertices"):
                num_edge_points = 16

                # Number of trianges = 2 * num_edge_points
                # one set for the base
                # one set for the edges
                #
                # Base is at origin, tip is at 1,0,0

                vertices = numpy.empty( 4 * 3*2*num_edge_points, dtype=numpy.float32)
                normals = numpy.empty( 3 * 3*4*num_edge_points, dtype=numpy.float32)

                # Make an array of phis to make sure identical floats show up
                #   where they're supposed to
                phis = numpy.arange(num_edge_points+1) * 2*math.pi / num_edge_points
                phis[num_edge_points] = 0.
                dphi = math.pi / num_edge_points

                # Endcap
                for i in range(num_edge_points):
                    vertices[4 * (3*i+0) : 4 * (3*i+0) + 4] = [0., 0., 0., 1.]
                    vertices[4 * (3*i+1) : 4 * (3*i+1) + 4] = [0., math.cos(phis[i]), math.sin(phis[i]), 1.]
                    vertices[4 * (3*i+2) : 4 * (3*i+2) + 4] = [0., math.cos(phis[i+1]), math.sin(phis[i+1]), 1.]
                    normals[3 * (3*i+0) : 3 * (3*i+0) + 3] = [-1., 0., 0.]
                    normals[3 * (3*i+1) : 3 * (3*i+1) + 3] = [-1., 0., 0.]
                    normals[3 * (3*i+2) : 3 * (3*i+2) + 3] = [-1., 0., 0.]

                # Edges
                off = 3*num_edge_points
                sqrt2 = math.sqrt(2.)
                for i in range(num_edge_points):
                    vertices[4 * (off + 3*i + 0) : 4 * (off + 3*i + 0) + 4] = [0., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    vertices[4 * (off + 3*i + 1) : 4 * (off + 3*i + 1) + 4] = [0., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    vertices[4 * (off + 3*i + 2) : 4 * (off + 3*i + 2) + 4] = [1., 0., 0., 1.]

                    
                    normals[3 * (off + 3*i + 0) : 3 * (off + 3*i + 0) + 3] = [1./sqrt2, math.cos(phis[i])/sqrt2,
                                                                              math.sin(phis[i])/sqrt2]
                    normals[3 * (off + 3*i + 1) : 3 * (off + 3*i + 1) + 3] = [1./sqrt2, math.cos(phis[i+1])/sqrt2,
                                                                              math.sin(phis[i+1])/sqrt2]
                    normals[3 * (off + 3*i + 2) : 3 * (off + 3*i + 2) + 3] = [1., 0., 0.]

                Cone._vertices = vertices
                Cone._normals = normals


    def __init__(self, radius=1., *args, **kwargs):
        super().__init__(*args, **kwargs)

        Cone.make_cone_vertices()

        self.vertexdata = Cone._vertices
        self.normaldata = Cone._normals
        self.num_triangles = len(self.vertexdata) // 12

        self.radius = radius

        sys.stderr.write("Made cone with radius {} and {} triangles.\n".format(radius, self.num_triangles))
        
        self.finish_init()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.scale = [self._scale[0], value, value]

# ======================================================================

class Arrow(GrObject):
    def __init__(self, shaftwidth=None, headwidth=None, headlength=None, fixedwidth=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        length = self.sx
        if shaftwidth is None:
            shaftwidth = 0.1 * length
        if headwidth is None:
            headwidth = 2 * shaftwidth
        if headlength is None:
            headlength = 3 * shaftwidth

        if not fixedwidth:
            headlength /= shaftwidth
            headwidth /= shaftwidth
            shaftwidth /= length

        self.fixedwidth = fixedwidth
        self.shaftwidth = shaftwidth
        self.headwidth = headwidth
        self.headlength = headlength

        self.make_arrow()

        self.finish_init()

    def make_arrow(self):
        # 16 triangles:
        #   2 for the base
        #   2 for each side (8 total)
        #   2 for the bottom of the head
        #   4 for the head
        self.num_triangles = 16
        self.vertexdata = numpy.empty(4 * 16 * 3, dtype=numpy.float32)
        self.normaldata = numpy.empty(3 * 16 * 3, dtype=numpy.float32)

        length = self.sx
        # Base
        if self.fixedwidth:
            shaftw = self.shaftwidth
            headw = self.headwidth
            headl = self.headlength
        else:
            shaftw = self.shaftwidth * length
            headw = self.headwidth * shaftw
            headl = self.headlength * shaftw

        # The length will get scaled by sx, so I have to renormalize
        # here.  (This is kind of messy, with sx having a special
        # meaning.  Side effect of inheritance, I guess.)
            
        headl /= length
        if (headl > 0.5): headl = 0.5
        shaftl = 1. - headl
            
        # sys.stderr.write("length={:.3f}, shaftl={:.3f}, shaftw={:.3f}, headw={:.3f}, headl={:.3f}\n"
        #                  .format(length, shaftl, shaftw, headw, headl))

        # Base
        self.vertexdata[0:3*2*4] = [0., -shaftw/2., shaftw/2., 1.,
                                    0., shaftw/2., -shaftw/2., 1.,
                                    0., shaftw/2., shaftw/2., 1.,
                                    
                                    0., -shaftw/2., shaftw/2., 1.,
                                    0., -shaftw/2., -shaftw/2., 1.,
                                    0., shaftw/2., -shaftw/2., 1.]
        self.normaldata[0:6*3] = [-1., 0., 0., -1., 0., 0., -1., 0., 0.,
                                  -1., 0., 0., -1., 0., 0., -1., 0., 0.]

        # Sides
        firstcorner = numpy.array([ [1, -1],  [1, 1],  [-1, 1], [-1, -1] ], dtype=numpy.int32)
        secondcorner = numpy.array([ [-1, -1], [1, -1], [1, 1],  [-1, 1] ], dtype=numpy.int32)
        normalvals = numpy.array([ [0., -1.], [1., 0.], [0., 1.], [-1., 0.] ],dtype=numpy.float32)
        off = 2
        for j in range(4):
            self.vertexdata[4 * (3 * (off + 2*j)) + 0 : 4 * (3 * (off + 2*j)) + 4] = [0.,
                                                                                      firstcorner[j, 0]*shaftw/2,
                                                                                      firstcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) + 4 : 4 * (3 * (off + 2*j)) + 8] = [0.,
                                                                                      secondcorner[j, 0]*shaftw/2,
                                                                                      secondcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) + 8 : 4 * (3 * (off + 2*j)) +12] = [shaftl,
                                                                                      secondcorner[j, 0]*shaftw/2,
                                                                                      secondcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) +12 : 4 * (3 * (off + 2*j)) +16] = [shaftl,
                                                                                      secondcorner[j, 0]*shaftw/2,
                                                                                      secondcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) +16 : 4 * (3 * (off + 2*j)) +20] = [shaftl,
                                                                                      firstcorner[j, 0]*shaftw/2,
                                                                                      firstcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) +20 : 4 * (3 * (off + 2*j)) +24] = [0.,
                                                                                      firstcorner[j, 0]*shaftw/2,
                                                                                      firstcorner[j, 1]*shaftw/2,
                                                                                      1.]
            for k in range(6):
                self.normaldata[3 * (3 * (off + 2*j) + k) + 0 :
                                3 * (3 * (off + 2*j) + k) + 3] = [0., normalvals[j,0], normalvals[j, 1]]
            
        # Base of head
        off = 10
        self.vertexdata[ 4 * (3*(off + 0)) : 4 * (3*(off + 2))] = [shaftl, -headw/2., headw/2., 1.,
                                                                   shaftl, headw/2., -headw/2., 1.,
                                                                   shaftl, headw/2., headw/2., 1.,
                                                                   
                                                                   shaftl, -headw/2., headw/2., 1.,
                                                                   shaftl, -headw/2., -headw/2., 1.,
                                                                   shaftl, headw/2., -headw/2., 1.]
        self.normaldata[ 3 * (3*(off + 0)) : 3 * (3*(off + 2))] = [-1., 0., 0., -1., 0., 0., -1., 0., 0.,
                                                                   -1., 0., 0., -1., 0., 0., -1., 0., 0.]
        

        # Head
        off = 12
        yzlen = headl / math.sqrt(headl*headl + (headw/2.)*(headw/2.))
        xlen = math.sqrt(1. - yzlen*yzlen)
        for j in range(4):
            self.vertexdata[ 4 * (3*(off+j)) :
                             4 * (3*(off+j + 1)) ] = [shaftl, firstcorner[j, 0]*headw/2, firstcorner[j, 1]*headw/2, 1.,
                                                      shaftl, secondcorner[j, 0]*headw/2, secondcorner[j, 1]*headw/2, 1.,
                                                      1., 0., 0., 1.]
            self.normaldata[ 3 * (3*(off+j)) :
                             3 * (3*(off +j + 1)) ] = [xlen, normalvals[j, 0]*yzlen, normalvals[j, 1]*yzlen,
                                                       xlen, normalvals[j, 0]*yzlen, normalvals[j, 1]*yzlen,
                                                       1., 0., 0]

    @GrObject.axis.setter
    def axis(self, value):
        GrObject.axis.fset(self, value)
        if self.fixedwidth:
            self.make_arrow()
            self.broadcast("update vertices")
        else:
            # I'm not completely happy about this, because the call to
            # fset(self, axis) and also setting scale means that
            # update_model_matrix gets called twice; we really should
            # have been able to skip the first one.  Add a parameter to
            # the GrObject.axis setter?  Seems janky
            length = math.sqrt(value[0]*value[0] + value[1]*value[1] + value[2]*value[2])
            self.scale = [length, length, length]
                
# ======================================================================
# length is the _base_ length of the spring.  The actual spring will be
# scaled up and down according to axis and scale as usual.
#
# Note!  Don't make getters and setters for properties that could change the number
# of triangles.  The underlying GrObject code assumes that once you've initialized,
# the number of triangles stays fixed.

class Helix(GrObject):
    def __init__(self, radius=1., coils=5., length=1., thickness=None,
                 num_edge_points=5, num_circ_points=8,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._length = length
        self._radius = radius
        if thickness is None:
            self._thickness = 0.05 * self._radius
        else:
            self._thickness = thickness
        self._coils = coils
        self._num_circ_points = int(num_circ_points)
        self._num_edge_points = int(num_edge_points)

        self._ncenters = int(math.floor(coils * num_circ_points+0.5)) + 1
        self.num_triangles = 2 * self._num_edge_points * (self._ncenters - 1)

        self.create_vertex_data()

        self.finish_init()

    def create_vertex_data(self):
        self.create_vertex_data_old()
        
    # This one is slower.... blah
    def create_vertex_data_new(self):
        # There are 2*num_edge_points triangles per center point (except for the last)
        # (I'm leaving the endcaps open)

        if self.vertexdata is None:
            self.vertexdata = numpy.empty( 4 * 3 * self.num_triangles, dtype=numpy.float32 )
            self.normaldata = numpy.empty( 3 * 3 * self.num_triangles, dtype=numpy.float32 )
        vertexdata = self.vertexdata
        normaldata = self.normaldata

        # psis are angles around the long local axis of the spring (for making the cross-section)
        psis = numpy.empty(self._num_edge_points, dtype=numpy.float32)
        psis[0:self._num_edge_points] = numpy.arange(self._num_edge_points) * 2.*math.pi / self._num_edge_points
        sinpsi = numpy.sin(psis, dtype=numpy.float32)
        cospsi = numpy.cos(psis, dtype=numpy.float32)

        # dphi is step around the circle from one point along the spring
        # to the next phi starts at 0 and increments by dphi for each
        # point along the spring the spring's axis is x; the points are
        # at x, z=r*cos(phi), y=r*sin(phi), where x starts at 0 for
        # point #0 and increments by dx (reaching length by point
        # #ncenters)
        dphi = 2.*math.pi / self._num_circ_points
        qdphi = numpy.array( [ -math.sin(dphi/2.), 0., 0., math.cos(dphi/2.) ] , dtype=numpy.float32)
        qinvdphi = qdphi.copy()
        qinvdphi[0:3] *= -1.
        
        # theta is the inclination of the spring
        dyz = 2 * self._radius * math.sin(dphi/2.)
        dx = self._length / (self._ncenters - 1.)
        l = math.sqrt(dyz*dyz + dx*dx)
        sintheta = dx/l
        costheta = dyz/l
        sintheta_2 = math.sqrt((1.-costheta)/2.)
        costheta_2 = math.sqrt((1.+costheta)/2.)
        
        
        # Make a base circle of points for the first point on the spring
        # Each subsequent point will be rotated by dphi around x, and
        # moved appropriately.  Start with a circle in the x-z plane,
        # and then rotate by tilt about -z (q is the quaternion of that
        # rotation).   nextcircle is the next one along
        #
        # (I use quaternion_multiply rather than quaterion_rotate here
        # to avoid recalculating qinv multiple times.)
        circle = numpy.empty( [self._num_edge_points, 4], dtype=numpy.float32 )
        circle[:, 0] = cospsi
        circle[:, 1] = 0.
        circle[:, 2] = sinpsi
        circle[:, 3] = 0.

        q = numpy.array( [0., 0., -sintheta_2, costheta_2] , dtype=numpy.float32 )
        qinv = q.copy()
        qinv[0:3] *= -1.

        for i in range(self._num_edge_points):
            circle[i] = quaternion_multiply(q, quaternion_multiply(circle[i], qinv))

        nextcircle = circle.copy()
        for i in range(self._num_edge_points):
            nextcircle[i] = quaternion_multiply(qdphi, quaternion_multiply(nextcircle[i], qinvdphi))
            
        # Build the first set of 2*num_edge_points triangles

        n1 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)
        n2 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)
        n3 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)
        n4 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)
        p1 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)
        p2 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)
        p3 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)
        p4 = numpy.empty( [self._num_edge_points, 4] , dtype=numpy.float32)

        n1[:] = circle
        n3[:] = nextcircle
        n2[:-1] = circle[1:]
        n2[-1] = circle[0]
        n4[:-1] = nextcircle[1:]
        n4[-1] = nextcircle[0]
        p1[:] = numpy.array( [ 0., 0., self._radius, 1.] , dtype=numpy.float32) + self._thickness * n1
        p2[:] = numpy.array( [ 0., 0., self._radius, 1.] , dtype=numpy.float32) + self._thickness * n2
        p3[:] = numpy.array( [dx, self._radius*numpy.sin(dphi),
                              self._radius*numpy.cos(dphi), 1.] , dtype=numpy.float32) + self._thickness * n3
        p4[:] = numpy.array( [dx, self._radius*numpy.sin(dphi),
                              self._radius*numpy.cos(dphi), 1.] , dtype=numpy.float32) + self._thickness * n4
        off = 0
        for i in range(self._num_edge_points):
            vertexdata[4 * (off + 3*0) + 0 : 4 * (off + 3*0) +  4] = p1[i]
            vertexdata[4 * (off + 3*0) + 4 : 4 * (off + 3*0) +  8] = p2[i]
            vertexdata[4 * (off + 3*0) + 8 : 4 * (off + 3*0) + 12] = p4[i]
            vertexdata[4 * (off + 3*1) + 0 : 4 * (off + 3*1) +  4] = p1[i]
            vertexdata[4 * (off + 3*1) + 4 : 4 * (off + 3*1) +  8] = p4[i]
            vertexdata[4 * (off + 3*1) + 8 : 4 * (off + 3*1) + 12] = p3[i]

            normaldata[3 * (off + 3*0) + 0 : 3 * (off + 3*0) +  3] = n1[i, 0:3]
            normaldata[3 * (off + 3*0) + 3 : 3 * (off + 3*0) +  6] = n2[i, 0:3]
            normaldata[3 * (off + 3*0) + 6 : 3 * (off + 3*0) +  9] = n4[i, 0:3]
            normaldata[3 * (off + 3*1) + 0 : 3 * (off + 3*1) +  3] = n1[i, 0:3]
            normaldata[3 * (off + 3*1) + 3 : 3 * (off + 3*1) +  6] = n4[i, 0:3]
            normaldata[3 * (off + 3*1) + 6 : 3 * (off + 3*1) +  9] = n3[i, 0:3]

            off += 6
            
        # Build the rest
        # I could make this more efficient by building just
        # one circle, and then offsetting subsequent circles by x
        for j in range(1, self._ncenters-1):
            newn3 = n1
            newn4 = n2
            newp3 = p1
            newp4 = p2
            for i in range(self._num_edge_points):
                newn3[i] = quaternion_multiply(qdphi, quaternion_multiply(n3[i], qinvdphi))
                newn4[i] = quaternion_multiply(qdphi, quaternion_multiply(n4[i], qinvdphi))
                newp3[i] = quaternion_multiply(qdphi, quaternion_multiply(p3[i], qinvdphi))
                newp4[i] = quaternion_multiply(qdphi, quaternion_multiply(p4[i], qinvdphi))
                newp3[i, 0] += dx
                newp4[i, 0] += dx
            p1 = p3
            p2 = p4
            p3 = newp3
            p4 = newp4
            n1 = n3
            n2 = n4
            n3 = newn3
            n4 = newn4
            for i in range(self._num_edge_points):
                vertexdata[4 * (off + 3*0) + 0 : 4 * (off + 3*0) +  4] = p1[i]
                vertexdata[4 * (off + 3*0) + 4 : 4 * (off + 3*0) +  8] = p2[i]
                vertexdata[4 * (off + 3*0) + 8 : 4 * (off + 3*0) + 12] = p4[i]
                vertexdata[4 * (off + 3*1) + 0 : 4 * (off + 3*1) +  4] = p1[i]
                vertexdata[4 * (off + 3*1) + 4 : 4 * (off + 3*1) +  8] = p4[i]
                vertexdata[4 * (off + 3*1) + 8 : 4 * (off + 3*1) + 12] = p3[i]

                normaldata[3 * (off + 3*0) + 0 : 3 * (off + 3*0) +  3] = n1[i, 0:3]
                normaldata[3 * (off + 3*0) + 3 : 3 * (off + 3*0) +  6] = n2[i, 0:3]
                normaldata[3 * (off + 3*0) + 6 : 3 * (off + 3*0) +  9] = n4[i, 0:3]
                normaldata[3 * (off + 3*1) + 0 : 3 * (off + 3*1) +  3] = n1[i, 0:3]
                normaldata[3 * (off + 3*1) + 3 : 3 * (off + 3*1) +  6] = n4[i, 0:3]
                normaldata[3 * (off + 3*1) + 6 : 3 * (off + 3*1) +  9] = n3[i, 0:3]

                off += 6

        # sys.stderr.write("Last off = {} ; self.num_triangles = {}\n".format(off, self.num_triangles))
        
        # with GLUTContext._threadlock:
        self.vertexdata = vertexdata
        self.normaldata = normaldata
                                        
        
        
    # This is a very expensive function, and for a spring
    #  it could be called a lot.  Optimize!
    # (I bet I could build a helix with really clever OpenGL instancing....)
    #
    # What I should probably do is create one circle's worth of normals,
    #  and then just use them repeatedly.  The only issue is when the number
    #  of coils isn't an integral number of helix segments.  (E.g., if
    #  num_circ_points = 5, and coils = 3.5, then the end of the coil is in between
    #  a couple of circles.)
    def create_vertex_data_old(self):
        # There are 2*num_edge_points triangles per center point (except for the last)
        # (I'm leaving the endcaps open)

        self.num_triangles = 2 * self._num_edge_points * (self._ncenters - 1)
        
        vertexdata = numpy.empty( 4 * 3 * self.num_triangles, dtype=numpy.float32 )
        normaldata = numpy.empty( 3 * 3 * self.num_triangles, dtype=numpy.float32 )

        # psis are angles around the long local axis of the spring (for making the cross-section)
        psis = numpy.arange(self._num_edge_points) * 2.*math.pi / self._num_edge_points
        sinpsi = numpy.sin(psis)
        cospsi = numpy.cos(psis)
        sinpsi1 = numpy.empty(len(sinpsi))
        sinpsi1[:-1] = sinpsi[1:]
        sinpsi1[-1] = sinpsi[0]
        cospsi1 = numpy.empty(len(cospsi))
        cospsi1[:-1] = cospsi[1:]
        cospsi1[-1] = cospsi[0]
        
        
        # phi tells us where we are along the spring
        dphi = 2*math.pi * self._coils / (self._ncenters - 1)
        phi = 0.
        nextphi = dphi

        x = 0.
        y = 0.
        z = self._radius
        rhat = numpy.array( [0., 0., 1., 0.] )

        nextx = self._length / float(self._ncenters - 1)
        nextz = math.cos(nextphi) * self._radius
        nexty = math.sin(nextphi) * self._radius
        lvec = numpy.array( [nextx - x, nexty - y, nextz - z, 0.] )
        lhat = lvec / math.sqrt(numpy.square(lvec).sum())
        yhat = numpy.array( [ rhat[1] * lhat[2] - rhat[2] * lhat[1],
                              rhat[2] * lhat[0] - rhat[0] * lhat[2],
                              rhat[0] * lhat[1] - rhat[1] * lhat[0], 0. ] )
        yhat /= math.sqrt(numpy.square(yhat).sum())
        
        n1 = numpy.empty( [self._num_edge_points, 4] )
        n2 = numpy.empty( [self._num_edge_points, 4] )
        n3 = numpy.empty( [self._num_edge_points, 4] )
        n4 = numpy.empty( [self._num_edge_points, 4] )
        p1 = numpy.empty( [self._num_edge_points, 4] )
        p2 = numpy.empty( [self._num_edge_points, 4] )
        p3 = numpy.empty( [self._num_edge_points, 4] )
        p4 = numpy.empty( [self._num_edge_points, 4] )

        off = 0
        for i in range(1, self._ncenters):
            lastphi = phi
            phi = nextphi
            nextphi = (i+1) * dphi
            lastx = x
            lasty = y
            lastz = z
            x = nextx
            y = nexty
            z = nextz
            lastrhat = rhat
            lastlhat = lhat
            lastyhat = yhat

            rhat = numpy.array( [0., math.sin(phi), math.cos(phi), 0.] )
            
            # This isn't quite right... I orient the thing perpendicular
            #  to the next cylinder in the chain, but really it should
            #  be the average of previous and next

            if i < self._ncenters-1 :
                nextx = (i+1) * self._length / float(self._ncenters - 1)
                nextz = math.cos(nextphi) * self._radius
                nexty = math.sin(nextphi) * self._radius
                lvec = numpy.array( [nextx - x, nexty - y, nextz - z, 0.] )
                lhat = lvec / math.sqrt(numpy.square(lvec).sum())
                yhat = numpy.array( [ rhat[1] * lhat[2] - rhat[2] * lhat[1],
                                      rhat[2] * lhat[0] - rhat[0] * lhat[2],
                                      rhat[0] * lhat[1] - rhat[1] * lhat[0], 0. ] )
            else:
                yhat = numpy.array( [ rhat[1] * lastlhat[2] - rhat[2] * lastlhat[1],
                                      rhat[2] * lastlhat[0] - rhat[0] * lastlhat[2],
                                      rhat[0] * lastlhat[1] - rhat[1] * lastlhat[0], 0. ] )

            yhat /= math.sqrt(numpy.square(yhat).sum())

            # sys.stderr.write("*** rhat = {}\n    lhat = {}\n    yhat = {}\n"
            #                  .format(rhat, lhat, yhat))
                
            n1[:] = sinpsi[:, numpy.newaxis] * lastrhat + cospsi[:, numpy.newaxis] * lastyhat
            n2[:] = sinpsi1[:, numpy.newaxis] * lastrhat + cospsi1[:, numpy.newaxis] * lastyhat
            n3[:] = sinpsi[:, numpy.newaxis] * rhat + cospsi[:, numpy.newaxis] * yhat
            n4[:] = sinpsi1[:, numpy.newaxis] * rhat + cospsi1[:, numpy.newaxis] * yhat

            p1[:] = n1 * self._thickness + numpy.array( [lastx, lasty, lastz, 1.] )
            p2[:] = n2 * self._thickness + numpy.array( [lastx, lasty, lastz, 1.] )
            p3[:] = n3 * self._thickness + numpy.array( [x, y, z, 1.] )
            p4[:] = n4 * self._thickness + numpy.array( [x, y, z, 1.] )
            
            
            for ipsi in range(self._num_edge_points):
                # I'm trying to make these right-handed, but I don't think it matters.
                # Plus, I may have thought about it wrong
                vertexdata[4 * (off + 3*0 + 0) : 4 * (off + 3*0 + 0) + 4] = p1[ipsi]
                vertexdata[4 * (off + 3*0 + 1) : 4 * (off + 3*0 + 1) + 4] = p2[ipsi]
                vertexdata[4 * (off + 3*0 + 2) : 4 * (off + 3*0 + 2) + 4] = p4[ipsi]
                vertexdata[4 * (off + 3*1 + 0) : 4 * (off + 3*1 + 0) + 4] = p1[ipsi]
                vertexdata[4 * (off + 3*1 + 1) : 4 * (off + 3*1 + 1) + 4] = p4[ipsi]
                vertexdata[4 * (off + 3*1 + 2) : 4 * (off + 3*1 + 2) + 4] = p3[ipsi]

                normaldata[3 * (off + 3*0 + 0) : 3 * (off + 3*0 + 0) + 3] = n1[ipsi,0:3]
                normaldata[3 * (off + 3*0 + 1) : 3 * (off + 3*0 + 1) + 3] = n2[ipsi,0:3]
                normaldata[3 * (off + 3*0 + 2) : 3 * (off + 3*0 + 2) + 3] = n4[ipsi,0:3]
                normaldata[3 * (off + 3*1 + 0) : 3 * (off + 3*1 + 0) + 3] = n1[ipsi,0:3]
                normaldata[3 * (off + 3*1 + 1) : 3 * (off + 3*1 + 1) + 3] = n4[ipsi,0:3]
                normaldata[3 * (off + 3*1 + 2) : 3 * (off + 3*1 + 2) + 3] = n3[ipsi,0:3]

                off += 6

        with GLUTContext._threadlock:
            self.vertexdata = vertexdata
            self.normaldata = normaldata

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self.create_vertex_data()
        self.broadcast("update vertices")

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.create_vertex_data()
        self.broadcast("update vertices")

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        self.create_vertex_data()
        self.broadcast("update vertices")
        
# ======================================================================
# A Curve is not a GrObject, even though some of the interface is the same

class Curve(object):
    def __init__(self, radius=0.01, maxpoints=50, color=color, points=None, num_edge_points=6, *args, **kwargs):
        self._position = numpy.array( [0., 0., 0.] )
        if points is not None:
            self._position = points[0]
        points = numpy.array(points)
        if len(points) > maxpoints: maxpoints = len(points)
        self.radius = radius
        self.color = color
        self.num_edge_points = num_edge_points

        self.pointbuffer = numpy.empty( [maxpoints, 3], dtype=numpy.float32 )
        self.cylbuffer = [None] * maxpoints

        for i in range(len(points)):
            self.pointbuffer[i, :] = points[i, :]
            if i < len(points)-1:
                axis = points[i+1] - points[i]
                self.cylbuffer[i] = Cylinder(position=points[i], axis=axis, color=self.color,
                                             radius=radius, num_edge_points=self.num_edge_points, *kargs, **kwargs)

        self.maxpoints = maxpoints
        self.nextpoint = len(points)
        if self.nextpoint > self.maxpoints:
            self.nextpoint = 0
        
    def add_point(self, point, *args, **kwargs):
        self.pointbuffer[self.nextpoint, :] = point[:]
        lastpoint = self.nextpoint - 1
        if lastpoint < 0 : lastpoint = self.maxpoints-1

        axis = self.pointbuffer[self.nextpoint, :] - self.pointbuffer[lastpoint, :]

        if self.cylbuffer[lastpoint] is not None:
            self.cylbuffer[lastpoint].position = self.pointbuffer[lastpoint]
            self.cylbuffer[lastpoint].axis = axis
        else:
            self.cylbuffer[lastpoint] = Cylinder(position=self.pointbuffer[lastpoint, :], axis=axis,
                                                 radius=self.radius, num_edge_points=self.num_edge_points,
                                                 color=self.color,
                                                 *args, **kwargs)

        self.nextpoint += 1
        if self.nextpoint >= self.maxpoints: self.nextpoint = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if len(value != 3):
            sys.stderr.write("ERROR, position must have 3 elements.")
            sys.exit(20)
        offset = numpy.array(position) - self._position
        self._position = numpy.array(position)
        # This is going to muck about with some uninitialized data, but it doesn't matter.
        self.pointbuffer += offset
        for i in range(self.maxpoints):
            if self.cylbuffer[i] is not None:
                self.cylbuffer[i].position = self.pointbuffer[i]

    
        
            
# ======================================================================

def main():
    # Big array of elongated boxes
    # sys.stderr.write("Making 100 elongated boxes.\n")
    # boxes = []
    # phases = []
    # n = 10
    # for i in range(n):
    #     for j in range (n):
    #         x = i*4./n - 2.
    #         y = j*4./n - 2.
    #         phases.append(random.random()*2.*math.pi)
    #         col = ( random.random(), random.random(), random.random() )
    #         boxes.append( Box(position=(x, y, 0.), axis=(1., -1., 1.), color=col, # color=color.red,
    #                           length=1.5, width=0.05, height=0.05))

    sys.stderr.write("Making boxes and peg and other things.\n")
    dobox1 = True
    dobox2 = True
    doball = True
    dopeg = True
    dopeg2 = True
    doblob = True
    doarrow = True
    dohelix = True

    # Make objects
    
    if dobox1:
        sys.stderr.write("Making box1.\n")
        box1 = Box(position=(-0.5, -0.5, 0), length=0.25, width=0.25, height=0.25, color=color.blue)
    if dobox2:
        sys.stderr.write("Making box2.\n")
        box2 = Box(position=( 0.5,  0.5, 0), length=0.25, width=0.25, height=0.25, color=color.red)

    if dopeg:
        sys.stderr.write("Making peg.\n")
        peg = Cylinder(position=(0., 0., 0.), radius=0.125, color=color.orange, num_edge_points=32)
        peg.axis = (0.5, 0.5, 0.5)
    if dopeg2:
        sys.stderr.write("Making peg2.\n")
        peg2 = Cylinder(position=(0., -0.25, 0.), radius=0.125, color=color.cyan, num_edge_points=6,
                        axis=(-0.5, 0.5, 0.5))
    if doblob:
        sys.stderr.write("Making blob.\n")
        blob = Ellipsoid(position=(0., 0., 0.), length=0.5, width=0.25, height=0.125, color=color.magenta)
        blob.axis = (-0.5, -0.5, 0.5)
    if doarrow:
        sys.stderr.write("Making arrow.\n")
        arrow = Arrow(position=(0., 0., 0.5), shaftwidth=0.05, headwidth = 0.1, headlength=0.2,
                      color=color.yellow, fixedwidth=True)
    
    if doball:
        sys.stderr.write("Making ball.\n")
        ball = Sphere(position= (2., 0., 0.), radius=0.5, color=color.green)
        # ball = Icosahedron(position = (2., 0., 0.), radius=0.5, color=color.green, flat=True, subdivisions=1)
    

    if dohelix:
        sys.stderr.write("Making helix.\n")
        helix = Helix(color = (0.5, 0.5, 0.), radius=0.2, thickness=0.05, length=2., coils=5,
                      num_circ_points=8, num_edge_points=5)
        
    # Updates
    
    theta = math.pi/4.
    phi = 0.
    phi2 = 0.
    fps = 30
    dphi = 2*math.pi/(4.*fps)

    GLUTContext._default_context.gl_version_info()

    first = True
    while True:

        # Animated angle
        phi += dphi
        if phi > 2.*math.pi:
            phi -= 2.*math.pi
        phi2 += dphi/3.7284317438
        if phi2 > 2.*math.pi:
            phi2 -= 2.*math.pi

        # Rotate all the elongated boxes
        # for i in range(len(boxes)):
        #     boxes[i].axis = numpy.array( [math.sin(theta)*math.cos(phi+phases[i]),
        #                                  math.sin(theta)*math.sin(phi+phases[i]),
        #                                  math.cos(theta)] )


        if doball:
            ball.x = 2.*math.cos(phi)
            if math.sin(phi)>0.:
                ball.rotate(dphi)
            else:
                ball.rotate(-dphi)

        if dobox2:
            q = numpy.array( [0., 0., -math.sin(math.pi/6.), math.cos(math.pi/6.)] )
            box2.position = quaternion_rotate(numpy.array( [ 2.*math.sin(phi2),
                                                             1.5*math.sin(phi),
                                                             1.5*math.cos(phi) ] ),
                                                           q )
            # box2.position = quaternion_multiply( numpy.array( [0., 0., -math.cos(math.pi/6), math.sin(math.pi/6)] ),
            #                                       quaternion_multiply( numpy.array( [ 2.*math.sin(phi2),
            #                                                                           1.5*math.sin(phi),
            #                                                                           1.5*math.cos(phi) ] ),
            #                                                             numpy.array( [0., 0., math.cos(math.pi/6),
            #                                                                             math.sin(math.pi/6)] )
            #                                                          ) )[0:3]
            
            if first:
                box2.interval = 5
                box2.retain = 50
                box2.make_trail = True
                first = False
    
        if doarrow:
            arrow.axis = [math.cos(phi) * (1. + 0.5*math.cos(phi)),
                          math.sin(phi) * (1. + 0.5*math.cos(phi)), 0.]
        
        if dohelix:
            helix.length = 2. + math.cos(phi)
            # helix.axis = [2. + math.cos(phi)/2., 0., 0.]
            

        rate(fps)




# ======================================================================

if __name__ == "__main__":
    main()

