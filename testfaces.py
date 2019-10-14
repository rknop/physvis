#!/usr/bin/python
# -*- coding: utf-8 -*-

import physvis as vis
import numpy

def main():
    l = 0.5
    verts = numpy.array( [ [-l, 0, -l], [ l, 0, -l], [-l, 0, l],
                           [-l, 0,  l], [ l, 0, -l], [ l, 0, l],
                           [-l, 0,  l], [ 0, 1.5*l, 0 ], [-l, 0, -l],
                           [ l, 0,  l], [ 0, 1.5*l, 0 ], [-l, 0,  l],
                           [ l, 0, -l], [ 0, 1.5*l, 0 ], [ l, 0,  l],
                           [-l, 0, -l], [ 0, 1.5*l, 0 ], [ l, 0, -l] ] )
    pyr = vis.faces(verts, color=vis.color.red)

    while (True):
        vis.rate(30)


# ======================================================================

if __name__ == "__main__":
    main()

