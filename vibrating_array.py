#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy
import time
from scipy.integrate import ode
from physvis import *
import visual_base

def dvals(t, vals, size, num):
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
                
    
def main():
    fps = 30.
    num = 3                        # of balls per axis (total = this cubed)
    size = 3                       # total size of array per axis
    l0 = size / (num-1)            # equil. length of most springs
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
    ballvals = numpy.zeros( (2, num, num, num, 3) )

    balls = []
    xsprings = []
    ysprings = []
    zsprings = []
    for k in range(num):
        balls.append([])
        xsprings.append([])
        ysprings.append([])
        if k < num-1:
            zsprings.append([])
        z = l0 * k - size/2.
        for j in range(num):
            y = l0 * j - size/2.
            balls[k].append([])
            xsprings[k].append([])
            if j < num-1:
                ysprings[k].append([])
            if k < num-1:
                zsprings[k].append([])
            for i in range(num):
                x = l0 * i - size/2.
                sys.stderr.write("Ball at {:.2f}, {:.2f}, {:.2f}\n".format(x, y, z))
                ballvals[0, k, j, i, 0] = x
                ballvals[0, k, j, i, 1] = y
                ballvals[0, k, j, i, 2] = z
                balls[k][j].append(sphere(radius = ballrad, pos = (x, y, z), color=ballcolor))
                if i < num-1:
                    if drawsprings:
                        xsprings[k][j].append(helix(radius=springrad, coils=springcoils, pos=(x, y, z),
                                                    num_circ_points=8, thickness=springthick,
                                                    axis=(l0, 0., 0.), length=1., color=springcolor))
                    else:
                        xsprings[k][j].append(cylinder(radius=springthick, pos=(x, y, z), color=springcolor,
                                                       axis=(l0, 0., 0.)))
                if j < num-1:
                    if drawsprings:
                        ysprings[k][j].append(helix(radius=springrad, coils=springcoils, pos=(x, y, z),
                                                    num_circ_points=8, thickness=springthick,
                                                    axis=(0., l0, 0.), length=1., color=springcolor))
                    else:
                        ysprings[k][j].append(cylinder(radius=springthick, pos=(x, y, z), color=springcolor,
                                                       axis=(0., l0, 0.)))
                if k < num-1:
                    if drawsprings:
                        zsprings[k][j].append(helix(radius=springrad, coils=springcoils, pos=(x, y, z),
                                                    num_circ_points=8, thickness=springthick,
                                                    axis=(0., 0., l0), length=1., color=springcolor))
                    else:
                        zsprings[k][j].append(cylinder(radius=springthick, pos=(x, y, z), color=springcolor,
                                                       axis=(0., 0., l0)))

    # offset a corner ball a bit to get things started
    ballvals[0, num-1, 0, 0, 0] += l0 / 4
    ballvals[0, num-1, 0, 0, 1] -= l0 / 6
    ballvals[0, num-1, 0, 0, 2] += l0 / 6

    oder = ode(dvals)
    oder.set_f_params(size, num)
    oder.set_integrator('vode', atol=1e-6, rtol=1e-9)
    ballvals.shape = [ 2 * num*num*num * 3 ]                               
    oder.set_initial_value(ballvals.copy(), 0.)
    ballvals.shape = [ 2, num, num, num, 3 ]
    t = 0.
    
    printfpsevery = 60
    nextprint = printfpsevery
    lasttime = time.perf_counter()
    while True:
        t += 1./fps

        # Make the array flat; I think scipy.integrate.ode needs this
        oder.y.shape = [ 2 * num*num*num * 3 ]                               
        oder.integrate(t)
        oder.y.shape = [ 2, num, num, num, 3 ]

        for k in range(num):
            for j in range(num):
                for i in range(num):
                    balls[k][j][i].pos = oder.y[0, k, j, i]
                    if i < num-1:
                        xsprings[k][j][i].pos = balls[k][j][i].pos
                        deltar = oder.y[0, k, j, i+1] - oder.y[0, k, j, i]
                        xsprings[k][j][i].axis = deltar
                    if j < num-1:
                        ysprings[k][j][i].pos = balls[k][j][i].pos
                        deltar = oder.y[0, k, j+1, i] - oder.y[0, k, j, i]
                        ysprings[k][j][i].axis = deltar
                    if k < num-1:
                        zsprings[k][j][i].pos = balls[k][j][i].pos
                        deltar = oder.y[0, k+1, j, i] - oder.y[0, k, j, i]
                        zsprings[k][j][i].axis = deltar
                        
        rate(fps)
        nextprint -= 1
        if nextprint <= 0 :
            nextprint = printfpsevery
            nexttime = time.perf_counter()
            sys.stderr.write("Effective main() fps = {}\n".format(printfpsevery / (nexttime - lasttime)))
            lasttime = nexttime
        
                                          
                

# ======================================================================

if __name__ == "__main__":
    main()
