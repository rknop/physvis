#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
from physvis import *

def main():
    x10 = -1.
    x20 = 1.
    l0 = math.fabs(x20-x10)
    k = 10.
    m = 1.
    v0 = 3.0
    fps = 30
    dt = 1./fps
    
    ball1 = sphere(radius=0.25, pos=(x10, 0, 0), color=color.red,
                   make_trail=True, interval=3, retain=50)
    ball1vel = vector( (0., v0, 0.) )
    ball2 = sphere(radius=0.25, pos=(x20, 0, 0), color=color.green,
                   make_trail=True, interval=3, retain=50)
    ball2vel = vector( (0., -v0, 0.) )

    spring = helix(radius=0.125, coils=8, thickness=0.025, pos=(x10, 0, 0),
                   axis=(x20-x10, 0, 0), color=(0.7, 0.6, 0))

    while True:
        spraxis = ball2.pos - ball1.pos
        sprlen = spraxis.mag
        sprhat = spraxis.norm()

        ball1vel += ( k/m * (sprlen - l0) * sprhat ) * dt
        ball2vel -= ( k/m * (sprlen - l0) * sprhat ) * dt
        ball1.pos += ball1vel * dt
        ball2.pos += ball2vel * dt
        spring.pos = ball1.pos
        spring.axis = ball2.pos - ball1.pos

        rate(fps)
        

# ======================================================================

if __name__ == "__main__":
    main()
