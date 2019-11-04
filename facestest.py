#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import sys
import qtgrcontext
import grcontext
import numpy
import visual_base as vb
import PyQt5.QtCore as qtcore
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui

wedgefaces = [ [+0.5, 0., -0.5],  [ 0.5, 0.,  0.5],   [-0.5, 0.,  0.5],
               [-0.5, 0.,  0.5],  [-0.5, 0., -0.5],   [ 0.5, 0., -0.5],
               [-0.5, 0.,  0.5],  [ 0.5, 0.,  0.5],   [-0.5, 1.,  0.5],
               [-0.5, 0., -0.5],  [ 0.5, 0., -0.5],   [-0.5, 1., -0.5],
               [-0.5, 0.,  0.5],  [-0.5, 1.,  0.5],   [-0.5, 0., -0.5],
               [-0.5, 1.,  0.5],  [-0.5, 1., -0.5],   [-0.5, 0., -0.5],
               [ 0.5, 0.,  0.5],  [ 0.5, 0., -0.5],   [-0.5, 1.,  0.5],
               [-0.5, 1.,  0.5],  [ 0.5, 0., -0.5],   [-0.5, 1., -0.5]
             ]

faces = vb.Faces(numpy.array(wedgefaces), color=vb.color.red)
scale = 0.5
faces.scale = (scale, 0.5, 1.)

cyl = vb.Cylinder(pos=(-0.5*scale, 0., 0.5), axis=(scale, 0., 0.), radius=0.05)

while(True):
   vb.rate(30)
    
