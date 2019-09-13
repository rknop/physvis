#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
from physvis import *

def main():
    fps = 30
    
    ctx1 = display()
    ctx2 = display()

    if (scene() is ctx1):
        print("scene is ctx1")
    if (scene() is ctx2):
        print("scene is ctx2")

    cube = box(width=0.5, height=0.25, length=0.75, color=color.red)
    ctx2.select()
    cube2 = box(width=1., height=0.175, length=0.5, color=color.green)
    cyl = cylinder(radius=0.5, axis=(1.5, 1.5, -0.25), color=color.blue)
    scene().select()
    spring = helix(radius=0.5, pos=(0.5, 0.5, 0.5), axis=(-1, -1, -1), thickness=0.05, color=color.magenta)
    
    counter = 0
    while True:
        counter +=1
        if (counter == 60):
            ctx1.width = 600
            ctx1.height = 600
            ctx1.title = "has been resized"
        rate(fps)
        

# ======================================================================

if __name__ == "__main__":
    main()
