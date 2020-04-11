#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import math
import argparse
import numpy
import physvis as vis
import qtgrcontext
from grcontext import GrContext
import PyQt5.QtCore as qtcore
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui


class TwoCurves(object):

    def __init__(self, endt=10., fps=100, printevery=2., *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.curve1 = vis.curve(color=vis.color.red, radius=0.1, retain=400)
        self.curve2 = vis.curve(color=vis.color.green, radius=0.1, retain=400)

        self.phiT1 = 5.
        self.phiT2 = 3.
        self.zT1 = 10.
        self.zT2 = 7.
        self.rT1 = 3.
        self.rT2 = 5.
        self.fps = fps
        self.t = 0.
        self.endt = endt
        
        self.printevery = printevery
        self.lastprint = time.perf_counter()
        self.framecount = 0

    def update(self, app):
        self.curve1.add_point( [ -2 + ( (1 + 0.8*math.sin(2*math.pi * self.t/self.rT1)) *
                                        math.cos(2*math.pi * self.t/self.phiT1) ) ,
                                 -math.cos(2*math.pi * self.t/self.zT1) ,
                                 (1 + 0.8*math.sin(2*math.pi * self.t/self.rT1)) *
                                 math.sin(2*math.pi * self.t/self.phiT1) ] )
        self.curve2.add_point( [ 2 + ( (1 + 0.8*math.sin(2*math.pi * self.t/self.rT2)) *
                                       math.cos(2*math.pi * self.t/self.phiT2) ) ,
                                 -math.cos(2*math.pi * self.t/self.zT2) ,
                                 (1 + 0.8*math.sin(2*math.pi * self.t/self.rT2)) *
                                 math.sin(2*math.pi * self.t/self.phiT2) ] )
        self.t += 1./self.fps
        if self.t >= self.endt:
            # Figure out the elegant way to really end
            if app is not None:
                app.closeAllWindows()
                app.exit()
            else:
                sys.exit()

        self.framecount += 1
        curtime = time.perf_counter()
        if curtime - self.lastprint > self.printevery:
            sys.stderr.write("Main loop {} fps\n".format( self.framecount / (curtime-self.lastprint) ))
            self.lastprint = curtime
            self.framecount = 0

def main():
    description = "Draw two growing curves"
    epilog = "Try setting fps to various different values.  If your graphics system syncs to the monitor's vblank (many of them do), this will limit how many calculation updates you can have per second to that rate.  At least *some* systems can disable this synchronization by setting the environment variable vblank_mode to 0; your OS and graphics card driver may have different ways to do this.  Try setting fps to different values (and using -g) and playing with this to see what you can effectively get."

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument("-t", "--endt", type=float, default=10., dest="endt",
                        help="Run until the simulation gets to this time in seconds (default: 10)")
    parser.add_argument("-q", "--qt", action="store_true", default=False, dest="qt",
                        help="Use the Qt backend (default: GLUT)")
    parser.add_argument("-f", "--fps", type=int, default=60, dest="fps",
                        help="Try to do this many updates per second (default: 60)")
    parser.add_argument("-p", "--printevery", type=float, default=2., dest="printevery",
                        help="Print updates per second every (roughtly) this many real seconds (default: 2)")
    parser.add_argument("-g", "--gfps", action="store_true", default=False, dest="gfps",
                        help="Print graphics frames per second")
    args = parser.parse_args()

    fps = args.fps
    endt = args.endt
    useqt = args.qt
    printevery = args.printevery

    GrContext.print_fps = args.gfps
    
    if useqt:
        app = qt.QApplication([])
        window = qt.QWidget()
        vbox = qt.QVBoxLayout()
        window.setLayout(vbox)

        wid = qtgrcontext.QtGrContext()
        vbox.addWidget(wid, 1)
        window.show()

    curver = TwoCurves(endt=endt, fps=fps, printevery=printevery)

    if useqt:
        mainlooptimer = qtcore.QTimer()
        mainlooptimer.timeout.connect(lambda : curver.update(app))
        mainlooptimer.start(1000./fps)
        app.exec_()
    else:
        while True:
            curver.update(None)
            vis.rate(fps)

    
# ======================================================================

if __name__ == "__main__":
    main()
