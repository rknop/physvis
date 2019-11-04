#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import time
import sys
import numpy
import queue
import threading
import qtgrcontext
import grcontext
from scipy.integrate import ode
import visual_base as vb
import PyQt5.QtCore as qtcore
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui

class roller(object):
    g = 9.8
    
    def __init__(self, qtapp):
        self.qtapp = qtapp
        self.initialized = False
        self.inittimer = qtcore.QTimer()
        self.inittimer.setSingleShot(True)
        self.inittimer.timeout.connect(lambda : self.qtinit())
        self.inittimer.start(1000. / 30.)

        self.theta = 30
        self.d = 1.
        self.nobjects = 2
        self.m = [0.25, 0.25]
        self.r = [0.1, 0.1]
        self.vis = ['Sphere', 'Sphere']
        self.I = [0.4, 0.4]
        self.tbot = ['', '']
        self.vbot = ['', '']
        self.rolling = [ False, False ]
        self.nrolling = 0
        
        self.vals = numpy.empty( 2*self.nobjects )
        
        self.objs = []
        for i in range(self.nobjects):
            self.objs.append(None)
        
        self.running = False
        self.t = 0.
        
        self.wedgefaces = numpy.array( [ [+0.5, 0., -0.5],  [ 0.5, 0.,  0.5],   [-0.5, 0.,  0.5],
                                         [-0.5, 0.,  0.5],  [-0.5, 0., -0.5],   [ 0.5, 0., -0.5],
                                         [-0.5, 0.,  0.5],  [ 0.5, 0.,  0.5],   [-0.5, 1.,  0.5],
                                         [-0.5, 0., -0.5],  [ 0.5, 0., -0.5],   [-0.5, 1., -0.5],
                                         [-0.5, 0.,  0.5],  [-0.5, 1.,  0.5],   [-0.5, 0., -0.5],
                                         [-0.5, 1.,  0.5],  [-0.5, 1., -0.5],   [-0.5, 0., -0.5],
                                         [ 0.5, 0.,  0.5],  [ 0.5, 0., -0.5],   [-0.5, 1.,  0.5],
                                         [-0.5, 1.,  0.5],  [ 0.5, 0., -0.5],   [-0.5, 1., -0.5] ] )

        # Calculation thread

        self.calcthread = threading.Thread(target=roller.runcalc, args=(self,))
        self.calcthread.start()

        self.things_to_run = queue.Queue()
        
    def qtinit(self):
        self.window = qt.QWidget()
        vbox = qt.QVBoxLayout()
        self.window.setLayout(vbox)

        fontdb = qtgui.QFontDatabase()
        if "DejaVu Serif" in fontdb.families():
            font = "DejaVu Serif"
        else:
            if not "Serif" in fontdb.families():
                raise Exception("Can't find Serif font.  Rob, fix this.")
            font = "Serif"
        sheet = "QWidget {{ font-family: \"{}\"; font-size: 18px }}".format(font)
        sys.stderr.write("Setting style sheet \"{}\"\n".format(sheet))
        self.window.setStyleSheet(sheet)

        
        self.physwid = qtgrcontext.QtGrContext()
        vbox.addWidget(self.physwid, 1)

        subwindow = qt.QWidget()
        vbox.addWidget(subwindow, 0)
        hbox = qt.QHBoxLayout()
        subwindow.setLayout(hbox)

        # First column : overall parameters
        
        buttoncol = qt.QWidget()
        hbox.addWidget(buttoncol)
        colvbox = qt.QVBoxLayout()
        buttoncol.setLayout(colvbox)

        hrow = qt.QWidget()
        colvbox.addWidget(hrow)
        hrowbox = qt.QHBoxLayout()
        hrow.setLayout(hrowbox)
        label = qt.QLabel("θ (⁰): ")
        hrowbox.addWidget(label, 0)
        self.thetabox = qt.QLineEdit()
        self.thetabox.setText(str(self.theta))
        hrowbox.addWidget(self.thetabox, 1)

        hrow = qt.QWidget()
        colvbox.addWidget(hrow)
        hrowbox = qt.QHBoxLayout()
        hrow.setLayout(hrowbox)
        label = qt.QLabel("d (m): ")
        hrowbox.addWidget(label, 0)
        self.dbox = qt.QLineEdit()
        self.dbox.setText(str(self.d))
        hrowbox.addWidget(self.dbox, 1)

        button = qt.QPushButton("Initialize")
        button.clicked.connect(lambda : self.initialize())
        colvbox.addWidget(button)
        button = qt.QPushButton("Roll 'Em")
        button.clicked.connect(lambda : self.roll())
        colvbox.addWidget(button)
        
        # Two columns of objects

        self.mbox = []
        self.rbox = []
        self.visbox = []
        self.Ibox = []
        self.color = []
        self.tbox = []
        self.vbox = []
        colors = [vb.color.red, vb.color.blue, vb.color.green,
                  vb.color.yellow, vb.color.cyan, vb.color.magenta,
                  vb.color.orange, vb.color.white]

        for i in range(self.nobjects):
            self.color.append(colors[i])
            
            buttoncol = qt.QWidget()
            hbox.addWidget(buttoncol)
            colvbox = qt.QVBoxLayout()
            buttoncol.setLayout(colvbox)

            hrow = qt.QWidget()
            colvbox.addWidget(hrow)
            hrowbox = qt.QHBoxLayout()
            hrow.setLayout(hrowbox)
            label = qt.QLabel("m₁ (kg): ")
            hrowbox.addWidget(label, 0)
            self.mbox.append(qt.QLineEdit())
            self.mbox[-1].setText(str(self.m[i]))
            hrowbox.addWidget(self.mbox[-1], 1)
        
            hrow = qt.QWidget()
            colvbox.addWidget(hrow)
            hrowbox = qt.QHBoxLayout()
            hrow.setLayout(hrowbox)
            label = qt.QLabel("r₁ (m): ")
            hrowbox.addWidget(label, 0)
            self.rbox.append(qt.QLineEdit())
            self.rbox[-1].setText(str(self.r[i]))
            hrowbox.addWidget(self.rbox[-1], 1)

            hrow = qt.QWidget()
            colvbox.addWidget(hrow)
            hrowbox = qt.QHBoxLayout()
            hrow.setLayout(hrowbox)
            label = qt.QLabel("visualize as: ")
            hrowbox.addWidget(label, 0)
            self.visbox.append(qt.QComboBox())
            self.visbox[-1].addItem("Sphere")
            self.visbox[-1].addItem("Cylinder")
            self.visbox[-1].addItem("Ring")
            self.visbox[-1].addItem("Cube")
            hrowbox.addWidget(self.visbox[-1], 1)

            hrow = qt.QWidget()
            colvbox.addWidget(hrow)
            hrowbox = qt.QHBoxLayout()
            hrow.setLayout(hrowbox)
            label = qt.QLabel("I =  ")
            hrowbox.addWidget(label, 0)
            self.Ibox.append(qt.QLineEdit())
            self.Ibox[-1].setText(str(self.I[i]))
            hrowbox.addWidget(self.Ibox[-1], 1)
            label = qt.QLabel("m r²")
            hrowbox.addWidget(label, 0)

            hrow = qt.QWidget()
            colvbox.addWidget(hrow)
            hrowbox = qt.QHBoxLayout()
            hrow.setLayout(hrowbox)
            label = qt.QLabel("t to base (s):  ")
            hrowbox.addWidget(label, 0)
            self.tbox.append(qt.QLabel())
            self.tbox[-1].setText(str(self.tbot[i]))
            hrowbox.addWidget(self.tbox[-1], 1)
            
            hrow = qt.QWidget()
            colvbox.addWidget(hrow)
            hrowbox = qt.QHBoxLayout()
            hrow.setLayout(hrowbox)
            label = qt.QLabel("v at base (m/s):  ")
            hrowbox.addWidget(label, 0)
            self.vbox.append(qt.QLabel())
            self.vbox[-1].setText(str(self.vbot[i]))
            hrowbox.addWidget(self.vbox[-1], 1)
            
        self.window.show()

        # Make our wedge

        self.wedge = vb.Faces(self.wedgefaces, color=(0.7, 0.4, 0.2),
                              scale = (self.d * math.cos(self.theta*math.pi/180.),
                                       self.d * math.sin(self.theta*math.pi/180.), self.nobjects))
        
        # Set up the regular update timeout

        self.updatetimer = qtcore.QTimer()
        self.updatetimer.timeout.connect(lambda : self.update())
        self.updatetimer.start(1000. / 30.)
        
    # ========================================

    def initialize(self):
        self.theta = float(self.thetabox.text())
        self.d = float(self.dbox.text())
        oldvis = self.vis.copy()
        for i in range(self.nobjects):
            self.m[i] = float(self.mbox[i].text())
            self.r[i] = float(self.rbox[i].text())
            self.I[i] = float(self.Ibox[i].text())
            self.vis[i] = self.visbox[i].currentText()
            
        sys.stderr.write("Theta = {}⁰, d={}m\n".format(self.theta, self.d))
        for i in range(self.nobjects):
            sys.stderr.write("Object {} is a {} with m={}kg, r={}m, and I={}mr²\n"
                             .format(i, self.vis[i], self.m[i], self.r[i], self.I[i]))

        self.running = False
        self.t = 0.
        self.vals[0:self.nobjects] = -self.d/2.
        self.vals[self.nobjects:] = 0.
            
        self.wedge.scale = (self.d * math.cos(self.theta*math.pi/180.),
                            self.d * math.sin(self.theta*math.pi/180.), self.nobjects)

        for i in range(self.nobjects):
            if self.objs[i] is not None:
                if self.vis[i] != oldvis[i]:
                    self.objs[i].visible = False
            if self.objs[i] is None or (not self.objs[i].visible):
                if self.vis[i] == "Sphere":
                    self.objs[i] = vb.Sphere(radius=self.r[i], color=self.color[i],
                                             pos = ( - self.d/2. * math.cos(self.theta*math.pi/180.),
                                                     self.r[i] + self.d*math.sin(self.theta*math.pi/180.),
                                                     -0.5 + i))
                elif self.vis[i] == "Cylinder":
                    self.objs[i] = vb.Cylinder(radius=self.r[i], color=self.color[i],
                                               pos = ( - self.d/2. * math.cos(self.theta*math.pi/180.),
                                                       self.r[i] + self.d*math.sin(self.theta*math.pi/180.),
                                                       -0.5 + i),
                                               axis = (0., 0., 0.25))
                elif self.vis[i] == "Ring":
                    self.objs[i] = vb.Ring(radius=0.9*self.r[i], color=self.color[i],
                                           pos = ( - self.d/2. * math.cos(self.theta*math.pi/180.),
                                                   self.r[i] + self.d*math.sin(self.theta*math.pi/180.),
                                                   -0.5 + i ),
                                           axis = (0., 0., 1.),
                                           thickness = self.r[i]/5.)
                elif self.vis[i] == "Cube":
                    self.objs[i] = vb.Box(length=2*self.r[i], width=2*self.r[i], height=2*self.r[i], color=self.color[i],
                                           pos = ( - self.d/2. * math.cos(self.theta*math.pi/180.),
                                                   self.r[i] + self.d*math.sin(self.theta*math.pi/180.),
                                                   -0.5 + i ) )
                else:
                    raise Exception("Unknown visualization shape {}".format(self.vis[i]))
                    
            if self.vis[i] == "Ring":
                self.objs[i].radius = 0.9*self.r[i]
                self.objs[i].thickness = self.r[i]/5.
            elif self.vis[i] == "Cube":
                self.objs[i].length = 2*self.r[i]
                self.objs[i].width = 2*self.r[i]
                self.objs[i].height = 2*self.r[i]
                self.objs[i].axis = ( 2*self.r[i]*math.cos(self.theta*math.pi/180.) ,
                                      -2*self.r[i]*math.sin(self.theta*math.pi/180.), 0. )
            else:
                self.objs[i].radius = self.r[i]
            self.objs[i].pos = ( - self.d/2. * math.cos(self.theta*math.pi/180.),
                                 self.r[i] + self.d*math.sin(self.theta*math.pi/180.),
                                 -0.5 + i)
            self.tbot[i] = ''
            self.vbot[i] = ''
            self.rolling[i] = True
        self.nrolling = self.nobjects
        self.update_widgets()

    def roll(self):
        self.running = True

    def update(self):
        try:
            while not self.things_to_run.empty():
                func = self.things_to_run.get()
                func()
        except queue.Empty:
            pass

    def update_widgets(self):
        for i in range(self.nobjects):
            if self.tbot[i] == "":
                self.tbox[i].setText("")
            else:
                self.tbox[i].setText("{:.3f}".format(self.tbot[i]))
            if self.vbot[i] == "":
                self.vbox[i].setText("")
            else:
                self.vbox[i].setText("{:.3f}".format(self.vbot[i]))
                

    def dvals_dt(self, t, vals):
        ders = numpy.zeros(self.nobjects*2)
        for i in range(self.nobjects):
            if self.rolling[i]:
                ders[i] = vals[i+self.nobjects]
                ders[self.nobjects+i] = roller.g * math.sin(self.theta*math.pi/180.) / (1.+self.I[i])
        return ders
    
    def runcalc(self):
        sys.stderr.write("In thread main.")

        wasrunning = False

        fps = 120
        while True:
            if self.running:
                if not wasrunning:
                    wasrunning = True
                    integrator = ode(lambda t, vals: self.dvals_dt(t, vals))
                    sys.stderr.write("Setting initial values: {}\n".format(self.vals))
                    integrator.set_initial_value(self.vals)
                
                self.t += 1./fps
                self.vals[:] = integrator.integrate(self.t)

                for i in range(self.nobjects):
                    if self.rolling[i]:
                        self.objs[i].x = self.vals[i] * math.cos(self.theta*math.pi/180.)
                        self.objs[i].y = self.r[i] + (self.d/2. - self.vals[i]) * math.sin(self.theta*math.pi/180.)
                        if self.objs[i].y <= self.r[i]:
                            sys.stderr.write("{} hit bottom\n".format(i))
                            self.rolling[i] = False
                            self.nrolling -= 1
                            sys.stderr.write("Reduced self.nrolling to {}\n".format(self.nrolling))
                            self.tbot[i] = self.t
                            self.vbot[i] = self.vals[self.nobjects + i]
                            self.things_to_run.put(lambda : self.update_widgets())

                if self.nrolling <= 0:
                    self.running = False
                            
            else:
                wasrunning = False

            vb.rate(fps)
                
        
# ======================================================================
        
def main():
    app = qt.QApplication([])

    win = roller(app)
    
    sys.stderr.write("Executing Qt app\n")
    app.exec_()



# ======================================================================

if __name__ == "__main__":
    main()
