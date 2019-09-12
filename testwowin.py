#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
from physvis import *
from visual_base import GLUTContext

def main():
    fps = 30
    
    ctx1 = GLUTContext(width=300, height=300, title="Context 1")
    ctx2 = GLUTContext(width=500, height=500, title="Context 2")
    # ctx2 = ctx1

    cube = box(width=0.5, height=0.25, length=0.75, color=color.red, context=ctx1)
    cube2 = box(width=1., height=0.175, length=0.5, color=color.green, context=ctx2)
    cyl = cylinder(radius=0.5, axis=(1.5, 1.5, -0.25), color=color.blue, context=ctx2)
    
    
    while True:
        rate(fps)
        

# ======================================================================

if __name__ == "__main__":
    main()
