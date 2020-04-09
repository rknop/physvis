#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy
import physvis as vis

def main():
    λx = 40
    λy = 50
    λz = 20
    Tx = 3.797
    Ty = 11.312
    Tz = 2.045
    n = 100
    points = numpy.empty( (n, 3) )
    points[:, 0] = 2. * numpy.sin( numpy.arange(n) * 2*math.pi/λx )
    points[:, 1] = 2. * numpy.sin( numpy.arange(n) * 2*math.pi/λy )
    points[:, 2] = 2. * numpy.sin( numpy.arange(n) * 2*math.pi/λz )
    basepoints = points.copy()
    
    curve = vis.curve(pos=points, color=vis.color.red, radius=0.1)

    fps = 30
    t = 0.
    while True:
        points = basepoints * numpy.array( [ numpy.cos( 2*math.pi * t / Tx ),
                                             numpy.cos( 2*math.pi * t / Ty ),
                                             numpy.cos( 2*math.pi * t / Tz ) ] )
        curve.points = points
        
        vis.rate(30)
        t += 1./fps
    
# ======================================================================

if __name__ == "__main__":
    main()

