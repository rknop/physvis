#/usr/bin/python3
# -*- coding: utf-8 -*-
#
# (c) 2019 by Rob Knop
#
# This file is part of physvis
#
# physvis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# physvis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with physvis.  If not, see <https://www.gnu.org/licenses/>.

"""This file has the code for the overall "context" (window or display)
in which we can put object collections.
"""

import sys
import math
import time
import queue
import threading
import ctypes

import itertools

import numpy

import OpenGL.GL as GL
import OpenGL.GLUT as GLUT

from rater import Rater
from physvis_observer import Subject, Observer
import quaternions as quat
import object_collection

# ======================================================================
# Context in which we could put objects.
#

class GrContext(Observer):

    """Encapsulates a window (or widget) and OpenGL context in which to draw.
    
    This class maintains the camera information.  Any object collections
    or shaders that are part of the collection must call the context's
    set_camera_posrot() method to update the camera position and
    orientation.  (ROB WRITE MORE.)
    
    Objects to be draw are in object collections.  An object collection
    adds itself to a context by calling the context's "add_collection"
    method.  (It can remove itself with "remove_collection".)  A
    collection must implement a "draw()" method that implements that
    collection's part of the OpenGL draw sequence.  An object collection
    must have a "shader" property with the two methods
    "set_camera_perspective(self)" and "set_camera_posrot(self)", which
    read the context's _perpmat (in the fist case) and _camrotate and
    _camtranslate (in the second case) properties and stick them as
    appropriate where needed in OpenGL uniforms.

    The following properties of this object define the camera:

      center — a point in 3d space that the camera is looking at.
      forward — a vector along which the camera looks.  (Normalized when set.)
      up — a vector that will be (as best possible) vertical on the display.  (Normalized when set.)
      fov — angular field of view (in radians) of the display
      range — size of an object at center that fills the display. 
                This is the radius of a sphere in world coordiantes
                whose angular size will be fov when it's right at the
                center position.  (Ideally just a single number, but you
                can also specify it as a 3-element vector, for
                historical reasons.)
 
    Camera position is internally calculated to be a distance of
    (range*tan(fov)) away from center along the opposite of the forward
    direction.

    There are several methods subclasses must implement:
      @property width
      @property height
      @property title
      update(self) — flag that the OpenGL display needs to be redrawn
      run_glcode(self,func) — schedule running func in the thread where all OpenGL happens

           Both update() and run_glcode() can be called from *any*
           thread, so the implementation should probably make queues or
           set flags or some such that will be read in whatever thread
           the context does its OpenGL drawing.

    Subclasses must set the following self variables before calligng super().__init__:
      _width
      _height
      _title
      (maybe some others?)

    """
    
    _default_instance = None
    _first_context = None
    
    print_fps = False
    
    @staticmethod
    def get_default_instance(*args, **kwargs):
        if GrContext._default_instance is None:
            # First one created makes itself default
            win = GLUTContext(*args, **kwargs)
        return GrContext._default_instance

    @staticmethod
    def get_new(qt=False, *args, **kwargs):
        if qt:
            raise Exception("Qt contexts not yet implemented")
        else:
            return GLUTContext(*args, **kwargs)
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.default_color = numpy.array([1., 1., 1., 1.])
        self.background_color = numpy.array([0., 0., 0., 0.])

        self._center = numpy.array( (0., 0., 0.) )    # What the camera looks at
        self._forward = numpy.array( (0., 0., -1.) )  # direction camera is facing
        self._fov = math.pi/3.0                       # camera field of view
        self._up = numpy.array( (0., 1., 0.) )        # a vector that will point up on the screen
        self._range = numpy.array( [3., 3., 3.] )     # Object of this radius will have angular size _fov

        self.determine_camera_matrices()
        
        self._clipnear = 0.1
        self._clipfar = 1000.

        self.determine_perspective_matrix()
        
        self.object_collections = []

        with Subject._threadlock:
            if GrContext._default_instance is None:
                # sys.stderr.write("Setting GrContext._default_instance to {}\n".format(self))
                GrContext._default_instance = self
                GrContext._first_context = self

    @property
    def foreground(self):
        return self.default_color

    @foreground.setter
    def foreground(self, val):
        val = numpy.array(val)
        if len(val) == 1 or len(val) == 3:
            self.default_color[0:3] = val
            self.default_color[3] = 1.
        elif len(val) == 4:
            self.default_color[:] = val
        else:
            raise Exception("foreground must have 1, 3, or 4 values")
        
    @property
    def background(self):
        return self.background_color

    @background.setter
    def foreground(self, val):
        val = numpy.array(val)
        if len(val) == 1 or len(val) == 3:
            self.background_color[0:3] = val
            self.background_color[3] = 1.
        elif len(val) == 4:
            self.background_color[:] = val
        else:
            raise Exception("foreground must have 1, 3, or 4 values")
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @width.setter
    def width(self, val):
        raise Exception("Subclasses must implement width property setter.")

    @height.setter
    def height(self, val):
        raise Exception("Subclasses must implement height property setter.")

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        if len(val) != 3:
            raise Exception("center needs 3 elements")
        self._center = numpy.array(val, dtype=float)
        self.run_glcode(lambda : self.update_cam_posrot_gl())

    @property
    def forward(self):
        return self._forward

    @forward.setter
    def forward(self, val):
        if len(val) != 3:
            raise Exception("forward needs 3 elements")
        self._forward = numpy.array(val, dtype=float)
        self._forward /= math.sqrt( self._forward[0]**2 + self._forward[1]**2 + self._forward[2]**2 )
        self.run_glcode(lambda : self.update_cam_posrot_gl())

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, val):
        if len(val) != 3:
            raise Exception("up needs 3 elements")
        self._up = numpy.array(val, dtype=float)
        self._up /= math.sqrt( self._up[0]**2 + self._up[1]**2 + self._up[2]**2 )
        self.run_glcode(lambda : self.update_cam_posrot_gl())

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, val):
        self._fov = float(val)
        self.run_glcode(lambda : self.update_cam_posrot_gl())
        self.run_glcode(lambda : self.update_cam_perspective_gl())
        
    @property
    def range(self):
        return self._range

    @range.setter
    def range(self, val):
        try:
            val = float(val)
            val = numpy.array( ( val, val, val ), dtype=float )
        except TypeError:
            val = numpy.array(val, dtype=float)
        if val.shape[0] != 3:
            raise Exception("range requires 1 or 3 values")
        self._range = val
        self.run_glcode(lambda : self.update_cam_posrot_gl())

    @property
    def scale(self):
        return 1./self._range

    @scale.setter
    def scale(self, val):
        try:
            val = float(val)
            val = numpy.array( ( 1/val, 1/val, 1/val ), dtype=float )
        except TypeError:
            val = numpy.array( val, dtype=float )
            val = 1./val
        if val.shape[0] != 3:
            raise Exception("scale requires 1 or 3 values")
        self._range = val
        self.run_glcode(lambda : self.update_cam_posrot_gl())
            
    def update(self):
        """Call this to flag the OpenGL renderer that things need to be redrawn."""
        raise Exception("GrContext subclasses need to implement update().")

    def run_glcode(self, func):
        """Call this to give a function that should be run in the GUI context."""
        raise Exception("GrContext subclasses need to implement run_glcode().")
    
    def select(self):
        """Makes this context the default context."""
        GrContext._default_instance = self

    def resize2d(self, width, height):
        """Callback that is called when window is resized."""
        # sys.stderr.write("In resize2d w/ size {} × {}\n".format(width, height))
        self._width = width
        self._height = height
        self.run_glcode(lambda : self.resize2d_gl())

    def resize2d_gl(self):
        GL.glViewport(0, 0, self._width, self._height)
        self.update_cam_perspective_gl()

    def determine_camera_matrices(self):
        center = self._center
        forward = numpy.array( self._forward )
        forward /= math.sqrt( forward[0]**2 + forward[1]**2 + forward[2]**2 )
        fov = self._fov
        up = numpy.array( self._up )
        up /= math.sqrt( up[0]**2 + up[1]**2 + up[2]**2 )
        rng = numpy.array( self._range )

        # I'm not dealing with non-equal range properly
        distback = rng[0] * math.tan(fov)

        # Is this right?
        self._position_of_camera = center - distback * forward

        self._camtranslate = numpy.matrix([[ 1., 0., 0., -center[0] ] ,
                                           [ 0., 1., 0., -center[1] ] ,
                                           [ 0., 0., 1., -center[2] ] ,
                                           [ 0., 0., 0., 1. ] ] )
                                       

        # Figure out if the camera needs to point in another direction

        costheta = -forward[2]      # -ẑ · forward
        if costheta < 1.-1e-8:
            if costheta > -1.+1e-6:
                rotabout = numpy.array( [ forward[1], -forward[0], 0. ] )   # -ẑ × forward
                rotabout /= math.sqrt( rotabout[0]**2 + rotabout[1]**2 + rotabout[2]**2 )
                costheta_2 = math.sqrt( (1+costheta) / 2. )
                sintheta_2 = math.sqrt( (1-costheta) / 2. )
                # Negative because we really rotate objects, not camera
                q = numpy.array( [ -sintheta_2 * rotabout[0], -sintheta_2 * rotabout[1],
                                   -sintheta_2 * rotabout[2], costheta_2 ] )
            else:
                # π about ŷ
                q = numpy.array( [ 0., 1., 0., 0. ] )
        else:
            q = numpy.array( [0., 0., 0., 1.] )

        # Rotate the up vector by this world rotation.  (up is a unit vector, so rotup will be too)

        rotup = quat.quaternion_rotate(up, q)

        # Project it into the camera (x-y) plane.

        rotup[2] = 0.
        magrotup = math.sqrt( rotup[0]**2 + rotup[1]**2 + rotup[2]**2 )
        cosphi = 0.
        if magrotup > 1e-8:
            # punt if the vector is too parallel to forward
            rotup /= magrotup
            if rotup[0] > 0:
                phiaxis = 1.
            else:
                phiaxis = -1.
            # Rotate this up projection to the y-axis
            cosphi = rotup[1]
            cosphi_2 = math.sqrt( (1+cosphi) / 2. )
            sinphi_2 = math.sqrt( (1-cosphi) / 2. )
            q = quat.quaternion_multiply( [ 0., 0., sinphi_2 * phiaxis, cosphi_2 ] , q )
            
        
        self._camrotate = numpy.array([[1 - 2*q[1]**2 - 2*q[2]**2, 2*q[0]*q[1] - 2*q[2]*q[3],
                                                   2*q[0]*q[2] + 2*q[1]*q[3], 0.],
                                       [2*q[0]*q[1] + 2*q[2]*q[3], 1 - 2*q[0]**2 - 2*q[2]**2,
                                                   2*q[1]*q[2] - 2*q[0]*q[3], 0.],
                                       [2*q[0]*q[2] - 2*q[1]*q[3], 2*q[1]*q[2] + 2*q[0]*q[3],
                                                   1 - 2*q[0]**2 - 2*q[1]**2, -distback],
                                       [ 0., 0., 0., 1.]], dtype=numpy.float32)

        # sys.stderr.write("_center = {}, _forward = {}, _up = {}\n".format(self._center,
        #                                                                   self._forward,
        #                                                                   self._up))
        # sys.stderr.write("_fov = {}, _range = {}\n".format(self._fov, self._range))
        # sys.stderr.write("distback = {}\n".format(distback))
        # sys.stderr.write("rotup = {}, costheta = {}, cosphi = {}\n".format(rotup, costheta, cosphi))
        # sys.stderr.write("camrotate:\n{}\n".format(self._camrotate))
        # sys.stderr.write("camtranslate:\n{}\n".format(self._camtranslate))
        
            
    def update_cam_posrot_gl(self):
        with Subject._threadlock:
            self.determine_camera_matrices()
            for collection in self.object_collections:
                collection.shader.set_camera_posrot()

    def determine_perspective_matrix(self):
        # Math from
        # https://www.opengl.org/discussion_boards/showthread.php/197893-View-and-Perspective-matrices
        #
        # See also
        #
        # http://www.songho.ca/opengl/gl_projectionmatrix.html
        #
        # I should understand this
        #
        # aspect is width/height

        s = 1.0/math.tan(self._fov/2.0)
        sx, sy = s / (self._width/self._height), s
        zz = (self._clipfar+self._clipnear)/(self._clipnear-self._clipfar)
        zw = 2*self._clipfar*self._clipnear/(self._clipnear-self._clipfar)
        self._perpmat = numpy.matrix([[sx, 0, 0, 0],
                                      [0, sy, 0, 0],
                                      [0, 0, zz, zw],
                                      [0, 0, -1, 0]], dtype=numpy.float32).T

    def update_cam_perspective_gl(self):
        self.determine_perspective_matrix()
        for collection in self.object_collections:
            collection.shader.set_camera_perspective()
    
    def gl_version_info(self):
        self.run_glcode(lambda : GrContext.do_gl_version_info())

    # It seems to be unhappy if you call this outside
    #  of a proper OpenGL Context.  Instead call
    #  the gl_version_info method of a GrContext instance.
    @staticmethod
    def do_gl_version_info():
        sys.stderr.write("OpenGL version: {}\n".format(GL.glGetString(GL.GL_VERSION)))
        sys.stderr.write("OpenGL renderer: {}\n".format(GL.glGetString(GL.GL_RENDERER)))
        sys.stderr.write("OpenGL vendor: {}\n".format(GL.glGetString(GL.GL_VENDOR)))
        sys.stderr.write("OpenGL shading language version: {}\n"
                         .format(GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)))

# ======================================================================
# ======================================================================
# ======================================================================
# ======================================================================

class GLUTContext(GrContext):
    """A GrContext that is a GLUT window.

    When you make one of these, it opens a window that has a OpenGL
    display in it.  No widgets, nothing else.

    """
    
    _class_init = False

    _instances = []

    _global_things_to_run = None
    
    # ======================================================================
    # Class methods

    @staticmethod
    def class_init(instance):

        with Subject._threadlock:
            if GLUTContext._class_init:
                return

            # sys.stderr.write("Doing GLUTContext class_init\n")

            GLUTContext._global_things_to_run = queue.Queue()
            
            # sys.stderr.write("Starting GLUT.GLUT thread...\n")
            GLUTContext.thread = threading.Thread(target=GLUTContext.thread_main, args=(instance,))
            GLUTContext.thread.daemon = True
            GLUTContext.thread.start()
            # sys.stderr.write("GrContext.thread.ident = {}\n".format(GrContext.thread.ident))
            # sys.stderr.write("Current thread ident = {}\n".format(threading.get_ident()))
            # sys.stderr.write("Main thread ident = {}\n".format(threading.main_thread().ident))

            # Class init finishes in thread_main(), and that's where it sets _class_init to True
            
    @staticmethod
    def class_idle():
        with Subject._threadlock:
            try:
                while not GLUTContext._global_things_to_run.empty():
                    func = GLUTContext._global_things_to_run.get()
                    func()
            except queue.Empty:
                pass
            for instance in GLUTContext._instances:
                instance.idle()
            
    @staticmethod
    def thread_main(instance):
        # sys.stderr.write("Starting thread_main\n")
        GLUT.glutInit(sys.argv)
        GLUT.glutInitContextVersion(3, 3)
        GLUT.glutInitContextFlags(GLUT.GLUT_FORWARD_COMPATIBLE)
        GLUT.glutInitContextProfile(GLUT.GLUT_CORE_PROFILE)
        GLUT.glutSetOption(GLUT.GLUT_ACTION_ON_WINDOW_CLOSE, GLUT.GLUT_ACTION_GLUTMAINLOOP_RETURNS)
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)

        # sys.stderr.write("Making default GLUT window.\n")
        GLUT.glutInitWindowSize(instance.width, instance.height)
        GLUT.glutInitWindowPosition(0, 0)
        instance.window = GLUT.glutCreateWindow(bytes(instance._title, encoding='UTF-8'))
        # freeglut on windows seems to need this here, even though I have a
        #  "for real" display func in gl_init
        GLUT.glutDisplayFunc(lambda : None)

        GLUT.glutIdleFunc(lambda : GLUTContext.class_idle())
        sys.stderr.write("Going into GLUT.GLUT main loop.\n")
        GLUTContext._class_init = True

        GLUT.glutMainLoop()

    def add_idle_func(func):
        self.idle_funcs.append(func)

    def remove_idle_func(func):
        self.idle_funcs = [x for x in GLUTContext.idle_funcs if x != func]

    def idle(self):
        if hasattr(self, "window") and self.window is not None:
            GLUT.glutSetWindow(self.window)
            with Subject._threadlock:
                try:
                    while not self.things_to_run.empty():
                        func = self.things_to_run.get()
                        func()
                except queue.Empty:
                    pass

                for func in self.idle_funcs:
                    func()


    # ======================================================================
    # Instance methods

    def __init__(self, width=500, height=400, title="PhysVis", *args, **kwargs):
        self._width = width
        self._height = height
        self._title = title

        super().__init__(*args, **kwargs)

        # sys.stderr.write("Starting GLUTContext.__init__\n")
        self.window_is_initialized = False
        self.framecount = 0

        self._mousex0 = 0.
        self._mousey0 = 0.

        self.idle_funcs = []
        self.things_to_run = queue.Queue()

        GLUTContext.class_init(self)

        # self.things_to_run.put(lambda : self.gl_init())
        GLUTContext._global_things_to_run.put(lambda : self.gl_init())

        while not self.window_is_initialized:
            time.sleep(0.1)

        # sys.stderr.write("Exiting GLUTContext.__init__\n")

    def gl_init(self):
        # sys.stderr.write("Starting GLUTContext.gl_init\n")
        if self is not GLUTContext._default_instance:
            # sys.stderr.write("Making a GLUT window.\n")
            GLUT.glutInitWindowSize(self._width, self._height)
            GLUT.glutInitWindowPosition(0, 0)
            self.window = GLUT.glutCreateWindow(bytes(self._title, encoding="UTF-8"))
        GLUT.glutSetWindow(self.window)
        GLUT.glutMouseFunc(lambda button, state, x, y : self.mouse_button_handler(button, state, x, y))
        GLUT.glutReshapeFunc(lambda width, height : self.resize2d(width, height))
        GLUT.glutDisplayFunc(lambda : self.draw())
        GLUT.glutVisibilityFunc(lambda state : self.window_visibility_handler(state))
        # Right now, the timer just prints FPS
        GLUT.glutTimerFunc(0, lambda val : self.timer(val), 0)
        GLUT.glutCloseFunc(lambda : self.cleanup())

        for coltype in object_collection.GLObjectCollection.collection_classes:
            self.object_collections.append(object_collection.GLObjectCollection.collection_classes[coltype](self))
        # self.object_collections.append(object_collection.SimpleObjectCollection(self))
        # self.object_collections.append(object_collection.CurveCollection(self))

        self.window_is_initialized = True
        GLUTContext._instances.append(self)
        # sys.stderr.write("Exiting gl_init\n")

    def update(self):
        GLUT.glutPostRedisplay()
        
    def run_glcode(self, func):
        # sys.stderr.write("Starting run_glcode\n")
        self.things_to_run.put(func)
        
    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @width.setter
    def width(self, val):
        self._width = val
        self.run_glcode(lambda : self.gottaresize())

    @height.setter
    def height(self, val):
        self._height = val
        self.run_glcode(lambda : self.gottaresize())

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, val):
        self._title = val
        self.run_glcode(lambda : self.gottasettitle())

    def gottaresize(self):
        GLUT.glutSetWindow(self.window)
        GLUT.glutReshapeWindow(self._width, self._height)

    def gottasettitle(self):
        GLUT.glutSetWindow(self.window)
        GLUT.glutSetWindowTitle(self._title)
                        
    def window_visibility_handler(self, state):
        if state != GLUT.GLUT_VISIBLE:
            return
        GLUT.glutSetWindow(self.window)
        with Subject._threadlock:
            GrContext._full_init = True
        GLUT.glutVisibilityFunc(None)

    def mouse_button_handler(self, button, state, x, y):
        if button == GLUT.GLUT_RIGHT_BUTTON:
            GLUT.glutSetWindow(self.window)

            if state == GLUT.GLUT_UP:
                # sys.stderr.write("RMB up:  forward={}\n  up={}\n".format(self._forward, self._up))
                # sys.stderr.write("         θ={:.2f}, φ={:.2f}, upθ={:.2f}, upφ={:.2f}\n"
                #                  .format(self._theta, self._phi, self._uptheta, self._upphi))
                GLUT.glutMotionFunc(None)

            elif state == GLUT.GLUT_DOWN:
                # sys.stderr.write("RMB down\n")
                self._mousex0 = x
                self._mousey0 = y
                self._origtheta = math.acos(-self._forward[1] / math.sqrt( self._forward[0]**2 +
                                                                           self._forward[1]**2 +
                                                                           self._forward[2]**2 ) )
                self._origphi = math.atan2(-self._forward[0], -self._forward[2])
                # sys.stderr.write("origθ = {:.2f}, origφ = {:.2f}\n".format(self._origtheta, self._origphi))
                GLUT.glutMotionFunc(lambda x, y : self.rmb_moved(x, y))

        if button == GLUT.GLUT_MIDDLE_BUTTON:
            GLUT.glutSetWindow(self.window)

            if state == GLUT.GLUT_UP:
                # sys.stderr.write("MMB up\n")
                GLUT.glutMotionFunc(None)

            elif state ==GLUT.GLUT_DOWN:
                # sys.stderr.write("MMB down\n")
                self._mousex0 = x
                self._mousey0 = y
                self._origrange = self._range
                GLUT.glutMotionFunc(lambda x, y : self.mmb_moved(x, y))

        if button == GLUT.GLUT_LEFT_BUTTON:
            GLUT.glutSetWindow(self.window)
            
            if state == GLUT.GLUT_UP:
                # sys.stderr.write("LMB up: self._center={}\n".format(self._center))
                GLUT.glutMotionFunc(None)

            if state == GLUT.GLUT_DOWN:
                # sys.stderr.write("LMB down\n")
                keys = GLUT.glutGetModifiers()
                if keys & GLUT.GLUT_ACTIVE_SHIFT:
                    self._mouseposx0 = x
                    self._mouseposy0 = y
                    self._origcenter = self._center
                    self._upinscreen = self._up - self._forward * ( numpy.sum(self._up*self._forward ) /
                                                                    math.sqrt( self._up[0]**2 +
                                                                               self._up[1]**2 +
                                                                               self._up[2]**2 ) )
                    self._upinscreen /= math.sqrt( self._upinscreen[0]**2 + self._upinscreen[1]**2 +
                                                   self._upinscreen[2]**2 )
                    self._rightinscreen = numpy.array( [ self._forward[1]*self._upinscreen[2] -
                                                            self._forward[2]*self._upinscreen[1],
                                                         self._forward[2]*self._upinscreen[0] -
                                                            self._forward[0]*self._upinscreen[2],
                                                         self._forward[0]*self._upinscreen[1] -
                                                            self._forward[1]*self._upinscreen[0] ] )
                    self._rightinscreen /= math.sqrt( self._rightinscreen[0]**2 + self._rightinscreen[1]**2 +
                                                      self._rightinscreen[2]**2 )
                                                        
                    GLUT.glutMotionFunc(lambda x, y : self.lmb_moved(x, y))
            
        if (state == GLUT.GLUT_UP) and ( button == 3 or button == 4):   # wheel up/down
            GLUT.glutSetWindow(self.window)

            if button == 3:
                self._range *= 0.9
            else:
                self._range *= 1.1
            self.update_cam_posrot_gl()


    def rmb_moved(self, x, y):
        dx = x - self._mousex0
        dy = y - self._mousey0
        theta = self._origtheta - dy * math.pi/2. / self._height
        if theta > math.pi:
            theta = math.pi
        if theta < 0.:
            theta = 0.
        phi = self._origphi - dx * math.pi / self._width
        # if phi < -math.pi:
        #     phi += 2.*math.pi
        # if phi > math.pi:
        #     phi -= 2.*math.pi
        self._forward = numpy.array( [ -math.sin(theta) * math.sin(phi),
                                       -math.cos(theta),
                                       -math.sin(theta) * math.cos(phi) ] )
        uptheta = theta - math.pi/2.
        upphi = phi
        if uptheta < 0.:
            uptheta = math.fabs(uptheta)
            upphi += math.pi
        self._up = numpy.array( [ math.sin(uptheta) * math.sin(upphi),
                                  math.cos(uptheta),
                                  math.sin(uptheta) * math.cos(upphi) ] )
        # self._up = numpy.array( [0., 1., 0.] )
        # sys.stderr.write("Moved from (θ,φ) = ({:.2f},{:.2f}) to ({:.2f},{:.2f})\n".
        #                  format(self._origtheta, self._origphi, theta, phi))
        # sys.stderr.write("Forward is now: {}\n".format(self._forward))
        self._theta = theta
        self._phi = phi
        self._uptheta = uptheta
        self._upphi = upphi
        self.update_cam_posrot_gl()


    def mmb_moved(self, x, y):
        dy = y - self._mousey0
        self._range = self._origrange * 10.**(dy/self._width)
        self.update_cam_posrot_gl()

    def lmb_moved(self, x, y):
        dx = x - self._mouseposx0
        dy = y - self._mouseposy0

        self._center = self._origcenter - self._rightinscreen * dx / self._width * self._range[0]
        self._center += self._upinscreen * dy / self._height * self._range[1]
        self.update_cam_posrot_gl()
        
    def receive_message(self, message, subject):
        sys.stderr.write("OMG!  Got message {} from subject {}, should do something!\n"
                         .format(message, subject))

    def cleanup(self):
        # sys.stderr.write("cleanup called in thread {}\n".format(threading.get_ident()))
        Rater.exit_whole_program()
        # I should do better than this:
        #  * actually clean up
        #  * think about multiple windows

    def timer(self, val):
        if GrContext.print_fps:
            sys.stderr.write("{} display fps: {}\n".format(self._title, self.framecount/2.))
        self.framecount = 0
        GLUT.glutTimerFunc(2000, lambda val : self.timer(val), 0)

    def draw(self):
        """The OpenGL draw routine."""
        
        GL.glClearColor(self.background_color[0], self.background_color[1],
                        self.background_color[2], self.background_color[3])
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        with Subject._threadlock:
            # sys.stderr.write("About to draw collections\n")
            for collection in self.object_collections:
                
                collection.draw()

                err = GL.glGetError()
                if err != GL.GL_NO_ERROR:
                    sys.stderr.write("Error {} drawing: {}\n".format(err, gluErrorString(err)))
                    sys.exit(-1)

            GLUT.glutSwapBuffers()
            # sys.stderr.write("Done drawing collections.\n")

            Rater.get().set()
            
        self.framecount += 1

    def add_object(self, obj):
        self.run_glcode( lambda : self.do_add_object(obj) )

    def remove_object(self, obj):
        self.run_glcode( lambda : self.do_remove_object(obj) )

    def do_add_object(self, obj):
        # See if any current collection will take it:
        for collection in self.object_collections:
            if collection.canyoutake(obj):
                collection.add_object(obj)
                return

        # If nobody took it, get a new collection

        newcollection = object_collection.GLObjectCollection.get_new_collection(obj, self)
        sys.stderr.write("CREATED a new collection for object type {}\n".format(newcollection.my_object_type))
        newcollection.add_object(obj)
        self.object_collections.append(newcollection)

    def remove_object(self, obj):
        for collection in self.object_collections:
            collection.remove_object(obj)

