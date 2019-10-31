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

        sys.stderr.write("Creating QtGrContext\n")

        fmt = qtgui.QSurfaceFormat()
        fmt.setMajorVersion(3)
        fmt.setMinorVersion(3)
        self.setFormat(fmt)

        self.window_is_initialized = False
        self.framecount = 0

        self._mousex0 = 0.
        self._mousey0 = 0.

        self.things_to_run = queue.Queue()
        self.qtimer = qtcore.QTimer()
        self.qtimer.timeout.connect(lambda : self.idlefunc())
        # I'm going to run my timer every 10 milliseconds; should I go full bore, or slower, or...?
        self.qtimer.start(10)

        if GrContext.print_fps:
            self.fpstimer = qtcore.QTimer()
            self.qtimer.timeout.connect(lambda : self.printfps())
            self.qtimer.start(2000)
        
    def initializeGL(self):
        sys.stderr.write("QtGrContext initializeGL\n")
        self.object_collections.append(object_collection.SimpleObjectCollection(self))
        self.object_collections.append(object_collection.CurveCollection(self))
        self.window_is_initialized = True

    def printfps(self):
        sys.stderr.write("{} display fps: {}\n".format(self._title, self.framecount/2.))
        self.framecount = 0
        
    def idlefunc(self):
        if not self.window_is_initialized:
            return
        try:
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


    
