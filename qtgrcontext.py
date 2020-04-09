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

import sys
import math
import queue
import PyQt5.QtCore as qtcore
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui
import OpenGL.GL as GL
import OpenGL.GLU as GLU
from grcontext import *

class QtGrContext(GrContext, qt.QOpenGLWidget):
    def __init__(self, width=500, height=400, title="PhysVis", *args, **kwargs):
        self._width = width
        self._height = height
        self._title = title

        super().__init__(*args, **kwargs)

        fmt = qtgui.QSurfaceFormat()
        fmt.setMajorVersion(3)
        fmt.setMinorVersion(3)
        self.setFormat(fmt)

        self.window_is_initialized = False
        self.framecount = 0

        self._mousex0 = 0.
        self._mousey0 = 0.
        self._leftmousemoving = False
        self._middlemousemoving = False
        self._rightmousemoving = False

        self.things_to_run = queue.Queue()
        self.qtimer = qtcore.QTimer()
        self.qtimer.timeout.connect(lambda : self.idlefunc())
        # I'm going to run my timer every 10 milliseconds; should I go full bore, or slower, or...?
        self.qtimer.start(10)
        
        if GrContext.print_fps:
            self.fpstimer = qtcore.QTimer()
            self.fpstimer.timeout.connect(lambda : self.printfps())
            self.fpstimer.start(2000)
        
    def initializeGL(self):
        self.object_collections.append(object_collection.SimpleObjectCollection(self))
        self.object_collections.append(object_collection.CurveCollection(self))
        self.window_is_initialized = True
        self.update()

    def printfps(self):
        sys.stderr.write("{} display fps: {}\n".format(self._title, self.framecount/2.))
        self.framecount = 0
        
    def idlefunc(self):
        if not self.window_is_initialized:
            return
        try:
            # Make sure that no rendering is happening while we do idle shit
            GL.glFinish()
            while not self.things_to_run.empty():
                func = self.things_to_run.get()
                func()
        except (queue.Empty):
            pass

    def update(self):
        qt.QOpenGLWidget.update(self)
        
    def run_glcode(self, func):
        self.things_to_run.put(func)
        
    def paintGL(self):
        if not self.window_is_initialized:
            return
        
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

            # Do I have to do anything to make QTGL double-buffer?

            Rater.get().set()
            
        self.framecount += 1

    def resizeGL(self, width, height):
        self.resize2d(width, height)                 # this is in the superclass

    def minimumSizeHint(self):
        return qtcore.QSize(100, 100)

    def sizeHint(self):
        return qtcore.QSize(640, 480)

    #========================================
    # Mouse handling
    #
    # A LOT of code is copied straight from GLUTContext.  I should
    # superclass it... or maybe move to only using Qt as my window
    # manager.

    # def mouseDoubleClickEvent(self, event):
    #     sys.stderr.write("Double click!\n")
    #     if self.isFullScreen():
    #         sys.stderr.write("Going normal\n")
    #         self.setParent(self.oldparent)
    #         self.resize(self.oldwindowsiae)
    #         self.overridwindowflags(self.oldflags)
    #         self.showNormal()
    #     else:
    #         sys.stderr.write("Going full screen\n")
    #         self.oldparent = self.parentWidget()
    #         self.oldwindowsize = self.size()
    #         self.oldflags = self.windowFlags()
    #         self.setParent(None)
    #         self.showFullScreen()
            
            
    def mousePressEvent(self, event):
        buts = event.buttons()
        mods = event.modifiers()
        x = event.pos().x()
        y = event.pos().y()
        
        if buts & qtcore.Qt.LeftButton:
            self._leftmousemoving = True
            if mods & qtcore.Qt.ShiftModifier:
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
                

        if buts & qtcore.Qt.RightButton:
            self._rightmousemoving = True
            self._mousex0 = x
            self._mousey0 = y
            self._origtheta = math.acos(-self.forward[1] / math.sqrt( self._forward[0]**2 +
                                                                      self._forward[1]**2 +
                                                                      self._forward[2]**2 ) )
            self._origphi = math.atan2(-self._forward[0], -self._forward[2])
            # sys.stderr.write("Right Press: θ = {:.3f}, φ = {:.3f}\n".format(self._origtheta, self._origphi))

        if buts & qtcore.Qt.MidButton:
            self._middlemousemoving = True
            self._mousex0 = x
            self._mousey0 = y
            self._origrange = self._range

    def mouseReleaseEvent(self, event):
        buts = event.buttons()
        if not buts & qtcore.Qt.LeftButton:
            self._leftmousemoving = False
        if not buts & qtcore.Qt.RightButton:
            self._rightmousemoving = False
        if not buts & qtcore.Qt.MidButton:
            self._middlemousemoving = False


    def mouseMoveEvent(self, event):
        mods = event.modifiers()
        x = event.pos().x()
        y = event.pos().y()
        
        if self._rightmousemoving:
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


        if self._leftmousemoving and (mods & qtcore.Qt.ShiftModifier) :
            dx = x - self._mouseposx0
            dy = y - self._mouseposy0

            self._center = self._origcenter - self._rightinscreen * dx / self._width * self._range[0]
            self._center += self._upinscreen * dy / self._height * self._range[1]
            self.update_cam_posrot_gl()

        if self._middlemousemoving:
            dy = y - self._mousey0
            self._range = self._origrange * 10.**(dy/self._width)
            self.update_cam_posrot_gl()


    def wheelEvent(self, event):
        dang = event.angleDelta().y()
        steps = abs( dang / 120. )
        if dang > 0:
            if self._range[0] > 1e-4:
                self._range *= (0.9 ** steps)
        elif dang < 0:
            if self._range[0] < 1e4:
                self._range *= (1.1 ** steps)
        # sys.stderr.write("Updating self._range to {}\n".format(self._range))
        self.update_cam_posrot_gl()
        
    # ========================================
    
    # These next two are 100% redundant with GLUTContext... maybe they should
    #   go into the superclass?
    
    def add_object(self, obj):
        self.run_glcode( lambda : self.do_add_object(obj) )

    def remove_object(self, obj):
        self.run_glcode( lambda : self.do_remove_object(obj) )

    def add_object(self, obj):

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


    
