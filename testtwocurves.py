#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import math
import numpy
import physvis as vis
import qtgrcontext
from grcontext import GrContext
import PyQt5.QtCore as qtcore
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui


QT = False

app = None
setup = False
phiT1 = 5.
phiT2 = 3.
zT1 = 10.
zT2 = 7.
rT1 = 3.
rT2 = 5.
curve1 = None
curve2 = None
fps = 60.
t = 0.
endt = 10.

lasttime = 0.

def mainloop(wid):
    global setup, curve1, curve2, fps, t
    global phiT1, phiT2, zT1, zT2, rT1, rT2
    global lasttime
    
    if not setup:
        curve1 = vis.curve(color=vis.color.red, radius=0.1, retain=400)
        curve2 = vis.curve(color=vis.color.green, radius=0.1, retain=400)
        setup = True
        
    curve1.add_point( [ -2 + (1+0.8*math.sin(2*math.pi*t/rT1)) * math.cos(2*math.pi*t/phiT1),
                        -math.cos(2*math.pi*t/zT1),
                        (1+0.8*math.sin(2*math.pi*t/rT1)) * math.sin(2*math.pi*t/phiT1) ] )
    curve2.add_point( [ 2 + (1+0.8*math.sin(2*math.pi*t/rT2)) * math.cos(2*math.pi*t/phiT2),
                        -math.cos(2*math.pi*t/zT2),
                        (1+0.8*math.sin(2*math.pi*t/rT2)) * math.sin(2*math.pi*t/phiT2) ] )
    t += 1./fps
    if t >= endt:
        if QT:
            app.closeAllWindows()
            app.exit()
        else:
            sys.exit()
    
    newtime = time.perf_counter()
    try:
        dtime = newtime - lasttime
        sys.stderr.write("Main loop {} fps\n".format(1./dtime))
    except(NameError):
        pass
    lasttime = newtime

def click(which, wid=None, *args, **kwargs):
    sys.stderr.write("which={}\n".format(which))
    sys.stderr.write("args:\n")
    for arg in args:
        sys.stderr.write("   {}\n".format(arg))
    sys.stderr.write("kwargs:\n")
    for arg in kwargs:
        sys.stderr.write("   {}={}\n".format(arg, kwargs[arg]))

    if wid is not None:
        sys.stderr.write("OpenGL Info:\n")
        wid.gl_version_info()


def main():
    global QT, fps, app

    GrContext.print_fps = True
    
    if QT:
        app = qt.QApplication([])
        window = qt.QWidget()
        vbox = qt.QVBoxLayout()
        window.setLayout(vbox)

        wid = qtgrcontext.QtGrContext()
        vbox.addWidget(wid, 1)
        window.show()

        mainlooptimer = qtcore.QTimer()
        mainlooptimer.timeout.connect(lambda : mainloop(wid) )
        mainlooptimer.start(1000./fps)

        app.exec_()
    else:
        while True:
            mainloop(None)
            vis.rate(fps)

    
# ======================================================================

if __name__ == "__main__":
    main()

