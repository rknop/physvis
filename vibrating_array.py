#!/usr/bin/python3
# -*- coding: utf-8 -*-

# A demo program for physvis
#
# vibrating_array.py is public domain
#
# physvis is (c) 2019 by Rob Knop and available under the GPL; see
# physvis.py and COPYING for more information.

import sys
import math
import numpy
import time
from scipy.integrate import ode
from physvis import *
import visual_base
import qtgrcontext
import grcontext
import PyQt5.QtCore as qtcore
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui

class vibarray(object):

    def dvals(self, t, vals, size, num):
        global dvalsdt
        m = 2.0
        sprconst = 100.0
        diagsprconst = 10.0
        l0 = size / (num-1)

        # More convenient views into vals
        vals.shape = [ 2, num, num, num, 3 ]
        r = vals[0]
        v = vals[1]

        dvalsdt = numpy.zeros( [2, num, num, num, 3] )
        drdt = dvalsdt[0]
        dvdt = dvalsdt[1]
        drdt[:] = v
        dvdt[:] = 0.

        # z springs
        deltar = r[1:, :, :, :] - r[:-1, :, :, :]
        magr = numpy.sqrt(numpy.square(deltar).sum(axis=3))
        rhat = deltar / magr[:, :, :, numpy.newaxis]
        dvdt[:-1, :, :, :] += sprconst/m * (magr-l0)[:, :, :, numpy.newaxis] * rhat
        dvdt[1: , :, :, :] -= sprconst/m * (magr-l0)[:, :, :, numpy.newaxis] * rhat

        # for i in range(num-1):
        #     deltar = r[i+1, :, :, :] - r[i, :, :, :]
        #     magr = numpy.sqrt(numpy.square(deltar).sum(axis=2))
        #     rhat = deltar[:, :, :] / magr[:, :, numpy.newaxis]

        #     dvdt[i,   :, :, :] += sprconst/m * (magr-l0)[:, :, numpy.newaxis] * rhat
        #     dvdt[i+1, :, :, :] -= sprconst/m * (magr-l0)[:, :, numpy.newaxis] * rhat

        # y springs
        deltar = r[:, 1:, :, :] - r[:, :-1, :, :]
        magr = numpy.sqrt(numpy.square(deltar).sum(axis=3))
        rhat = deltar / magr[:, :, :, numpy.newaxis]
        dvdt[:, :-1, :, :] += sprconst/m * (magr-l0)[:, :, :, numpy.newaxis] * rhat
        dvdt[:, 1: , :, :] -= sprconst/m * (magr-l0)[:, :, :, numpy.newaxis] * rhat

        # for i in range(num-1):
        #     deltar = r[:, i+1, :, :] - r[:, i, :, :]
        #     magr = numpy.sqrt(numpy.square(deltar).sum(axis=2))
        #     rhat = deltar[:, :, :] / magr[:, :, numpy.newaxis]

        #     dvdt[:, i  , :, :] += sprconst/m * (magr-l0)[:, :, numpy.newaxis] * rhat
        #     dvdt[:, i+1, :, :] -= sprconst/m * (magr-l0)[:, :, numpy.newaxis] * rhat

        # x springs
        deltar = r[:, :, 1:, :] - r[:, :, :-1, :]
        magr = numpy.sqrt(numpy.square(deltar).sum(axis=3))
        rhat = deltar / magr[:, :, :, numpy.newaxis]
        dvdt[:, :, :-1, :] += sprconst/m * (magr-l0)[:, :, :, numpy.newaxis] * rhat
        dvdt[:, :, 1: , :] -= sprconst/m * (magr-l0)[:, :, :, numpy.newaxis] * rhat

        # for i in range(num-1):
        #     deltar = r[:, :, i+1, :] - r[:, :, i, :]
        #     magr = numpy.sqrt(numpy.square(deltar).sum(axis=2))
        #     rhat = deltar[:, :, :] / magr[:, :, numpy.newaxis]

        #     dvdt[:, :, i  , :] += sprconst/m * (magr-l0)[:, :, numpy.newaxis] * rhat
        #     dvdt[:, :, i+1, :] -= sprconst/m * (magr-l0)[:, :, numpy.newaxis] * rhat


        # # diagonal springs
        # ldiag0 = math.sqrt(3.) * l0
        # deltar = r[1:, 1:, 1:, :] - r[:-1, :-1, :-1, :]
        # magr = numpy.sqrt(numpy.square(deltar).sum(axis=3))
        # rhat = deltar / magr[:, :, :, numpy.newaxis]
        # dvdt[:-1, :-1, :-1, :] += diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat
        # dvdt[1:,  1:,  1:,  :] -= diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat

        ldiag0 = math.sqrt(2.) * l0
        # diagonal springs xy
        deltar = r[:, 1:, 1:, :] - r[:, :-1, :-1, :]
        magr = numpy.sqrt(numpy.square(deltar).sum(axis=3))
        rhat = deltar / magr[:, :, :, numpy.newaxis]
        dvdt[:, :-1, :-1, :] += diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat
        dvdt[:, 1:,  1:,  :] -= diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat

        # diagonal springs xz
        deltar = r[1:, :, 1:, :] - r[:-1, :, :-1, :]
        magr = numpy.sqrt(numpy.square(deltar).sum(axis=3))
        rhat = deltar / magr[:, :, :, numpy.newaxis]
        dvdt[:-1, :, :-1, :] += diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat
        dvdt[1:, :,  1:,  :] -= diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat

        # diagonal springs yz
        deltar = r[1:, 1:, :, :] - r[:-1, :-1, :, :]
        magr = numpy.sqrt(numpy.square(deltar).sum(axis=3))
        rhat = deltar / magr[:, :, :, numpy.newaxis]
        dvdt[:-1, :-1, :, :] += diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat
        dvdt[1:,  1:,  :, :] -= diagsprconst/m * (magr-ldiag0)[:, :, :, numpy.newaxis] * rhat

        vals.shape = [ 2*num*num*num*3 ]
        dvalsdt.shape = [ 2*num*num*num*3 ]
        return dvalsdt
                
    def __init__(self):
        super().__init__()
        
        self.fps = 30.
        self.num = 4                   # of balls per axis (total = this cubed)
        size = 3                       # total size of array per axis
        l0 = size / (self.num-1)            # equil. length of most springs
        ldiag0 = math.sqrt(2) * l0     # equil. length of diagonal spring

        visual_base._print_fps = True

        ballrad = 0.2
        springrad = 0.05
        springthick = 0.02
        springcoils = 5
        ballcolor = (1., 0., 0.)
        springcolor = (0.5, 0.5, 0.)

        drawsprings = True

        # Store the data we'll use for the solution in numpy arrays, so we can use scipy.integrate.ode.
        self.ballvals = numpy.zeros( (2, self.num, self.num, self.num, 3) )

        self.balls = []
        self.xsprings = []
        self.ysprings = []
        self.zsprings = []
        for k in range(self.num):
            self.balls.append([])
            self.xsprings.append([])
            self.ysprings.append([])
            if k < self.num-1:
                self.zsprings.append([])
            z = l0 * k - size/2.
            for j in range(self.num):
                y = l0 * j - size/2.
                self.balls[k].append([])
                self.xsprings[k].append([])
                if j < self.num-1:
                    self.ysprings[k].append([])
                if k < self.num-1:
                    self.zsprings[k].append([])
                for i in range(self.num):
                    x = l0 * i - size/2.
                    sys.stderr.write("Ball at {:.2f}, {:.2f}, {:.2f}\n".format(x, y, z))
                    self.ballvals[0, k, j, i, 0] = x
                    self.ballvals[0, k, j, i, 1] = y
                    self.ballvals[0, k, j, i, 2] = z
                    self.balls[k][j].append(sphere(radius = ballrad, pos = (x, y, z), color=ballcolor))
                    if i < self.num-1:
                        if drawsprings:
                            self.xsprings[k][j].append(helix(radius=springrad, coils=springcoils, pos=(x, y, z),
                                                             num_circ_points=8, thickness=springthick,
                                                             axis=(l0, 0., 0.), length=1., color=springcolor))
                        else:
                            self.xsprings[k][j].append(cylinder(radius=springthick, pos=(x, y, z), color=springcolor,
                                                                axis=(l0, 0., 0.)))
                    if j < self.num-1:
                        if drawsprings:
                            self.ysprings[k][j].append(helix(radius=springrad, coils=springcoils, pos=(x, y, z),
                                                             num_circ_points=8, thickness=springthick,
                                                             axis=(0., l0, 0.), length=1., color=springcolor))
                        else:
                            self.ysprings[k][j].append(cylinder(radius=springthick, pos=(x, y, z), color=springcolor,
                                                                axis=(0., l0, 0.)))
                    if k < self.num-1:
                        if drawsprings:
                            self.zsprings[k][j].append(helix(radius=springrad, coils=springcoils, pos=(x, y, z),
                                                             num_circ_points=8, thickness=springthick,
                                                             axis=(0., 0., l0), length=1., color=springcolor))
                        else:
                            self.zsprings[k][j].append(cylinder(radius=springthick, pos=(x, y, z), color=springcolor,
                                                           axis=(0., 0., l0)))

        # offset a corner ball a bit to get things started
        self.ballvals[0, self.num-1, 0, 0, 0] += l0 / 4
        self.ballvals[0, self.num-1, 0, 0, 1] -= l0 / 6
        self.ballvals[0, self.num-1, 0, 0, 2] += l0 / 6

        self.oder = ode(self.dvals)
        self.oder.set_f_params(size, self.num)
        self.oder.set_integrator('vode', atol=1e-6, rtol=1e-9)
        self.ballvals.shape = [ 2 * self.num*self.num*self.num * 3 ]                               
        self.oder.set_initial_value(self.ballvals.copy(), 0.)
        self.ballvals.shape = [ 2, self.num, self.num, self.num, 3 ]
        self.t = 0.

        self.printfpsevery = 60
        self.nextprint = self.printfpsevery
        self.lasttime = time.perf_counter()

    def update(self):
        self.t += 1./self.fps

        # Make the array flat; I think scipy.integrate.ode needs this
        self.oder.y.shape = [ 2 * self.num*self.num*self.num * 3 ]                               
        self.oder.integrate(self.t)
        self.oder.y.shape = [ 2, self.num, self.num, self.num, 3 ]

        for k in range(self.num):
            for j in range(self.num):
                for i in range(self.num):
                    self.balls[k][j][i].pos = self.oder.y[0, k, j, i]
                    if i < self.num-1:
                        self.xsprings[k][j][i].pos = self.balls[k][j][i].pos
                        deltar = self.oder.y[0, k, j, i+1] - self.oder.y[0, k, j, i]
                        self.xsprings[k][j][i].axis = deltar
                    if j < self.num-1:
                        self.ysprings[k][j][i].pos = self.balls[k][j][i].pos
                        deltar = self.oder.y[0, k, j+1, i] - self.oder.y[0, k, j, i]
                        self.ysprings[k][j][i].axis = deltar
                    if k < self.num-1:
                        self.zsprings[k][j][i].pos = self.balls[k][j][i].pos
                        deltar = self.oder.y[0, k+1, j, i] - self.oder.y[0, k, j, i]
                        self.zsprings[k][j][i].axis = deltar

        self.nextprint -= 1
        if self.nextprint <= 0 :
            self.nextprint = self.printfpsevery
            self.nexttime = time.perf_counter()
            sys.stderr.write("Effective main() fps = {}\n"
                             .format(self.printfpsevery / (self.nexttime - self.lasttime)))
            self.lasttime = self.nexttime
        
# ======================================================================

def main():
    grcontext.GrContext.print_fps = True
    
    QT = False
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "qt":
            QT = True
    if QT:
        app = qt.QApplication([])
        window = qt.QWidget()
        vbox = qt.QVBoxLayout()
        window.setLayout(vbox)

        wid = qtgrcontext.QtGrContext()
        vbox.addWidget(wid, 1)
        window.show()

    solid = vibarray()

    if QT:
        mainlooptimer = qtcore.QTimer()
        mainlooptimer.timeout.connect(solid.update)
        mainlooptimer.start(1000./solid.fps)
        app.exec_()
    else:    
        while True:
            rate(solid.fps)
            solid.update()

# ======================================================================

if __name__ == "__main__":
    main()
