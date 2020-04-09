#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import physvis as vis

def main():
    fps = 30.
    endt = 3

    box = vis.box()
    
    t = 0
    while True:
        newtime = time.perf_counter()
        try:
            dtime = newtime - lasttime
            sys.stderr.write("fps: {}\n".format(1./dtime))
        except(NameError):
            pass
        lasttime = newtime
        vis.rate(fps)
        t += 1./fps
        if t >= endt:
            return
        
# ======================================================================

if __name__ == "__main__":
    main()
