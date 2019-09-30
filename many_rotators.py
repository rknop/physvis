#/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import random
from physvis import *

peg = cylinder(pos=(0., 0., 0.), radius=0.125, color=color.orange, num_edge_points=32)
peg.axis = (0.5, 0.5, 0.5)

n = 10
sys.stderr.write("Making {} elongated boxes.\n".format(n*n))
boxes = []
phases = []
for i in range(n):
    for j in range (n):
        x = i*4./n - 2.
        y = j*4./n - 2.
        phases.append(random.random()*2.*math.pi)
        col = ( random.random(), random.random(), random.random() )
        boxes.append( box(pos=(x, y, 0.), axis=(1., -1., 1.), color=col,
                          length=1.5, width=0.05, height=0.05))


_print_fps = True
printfpsevery = 30
lasttime = time.perf_counter()
nextprint = printfpsevery
fps = 30
dphi = 2*math.pi/(4.*fps)
phi = 0.
theta = math.pi/4.
st = math.sin(theta)
ct = math.cos(theta)

while True:

    # Animated angle
    phi += dphi
    if phi > 2.*math.pi:
        phi -= 2.*math.pi

    # Rotate all the elongated boxes
    for i in range(len(boxes)):
        boxes[i].axis = [st * math.cos(phi+phases[i]),
                         st * math.sin(phi+phases[i]),
                         ct ]

    rate(fps)
    nextprint -= 1
    if nextprint <= 0 :
        nextprint = printfpsevery
        nexttime = time.perf_counter()
        sys.stderr.write("Effective main() fps = {}\n".format(printfpsevery / (nexttime - lasttime)))
        lasttime = nexttime
