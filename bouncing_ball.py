#!/usr/bin/python3
# -*- coding: utf-8 -*-

# A demo program for physvis
#
# bouncing_ball.py is public domain
#
# physvis is (c) 2019 by Rob Knop and available under the GPL; see
# physvis.py and COPYING for more information.

import math
from physvis import *

def main():
    g = 9.8
    r = 0.25
    y0 = 1.8
    groundpos = -2
    fps = 30
    
    ball = sphere(radius=r, pos=(0, y0, 0), color=color.red)
    ground = box(width=5, length=5, height=0.2, pos=(0, groundpos-0.1, 0), color=color.green)
    ballv = 0.

    # (Ignore this comment if you just want an example of how to use
    # physvis.)
    #
    # Something to note: the ball loses energy at each bounce;
    # eventually, it sits on the ground, and then slowly sinks through
    # it.  This is a result of the numerical algorithm, which always
    # lets gravity pull the ball down one time step, even if it's
    # already on the ground.
    #
    # If you put the ball.y += ballv*dt statement *after* the if block,
    # the ball gains energy each bounce.
    #
    # (Doing this right requires handling the bounce in a more
    # sophisticated manner, e.g. by having a spring that gets compressed
    # when the ball's y position is low enough.  Having a smaller dt
    # would help too.)
    
    dt = 1./fps
    while True:
        ballv -= g * dt
        ball.y += ballv * dt
        if ball.y <= groundpos:
            ballv = math.fabs(ballv)

        rate(fps)
        

# ======================================================================

if __name__ == "__main__":
    main()
