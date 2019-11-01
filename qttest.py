#!/usr/bin/python3
# -*- coding: utf-8 -*-

import time
import sys
import qtgrcontext
import grcontext
import visual_base as vb
import PyQt5.QtCore as qtcore
import PyQt5.QtWidgets as qt
import PyQt5.QtGui as qtgui

def mainloop(wid):
    try:
        setup
    except:
        setup = True
        box = vb.Box(context=wid, color=vb.color.red)
        
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
    grcontext.GrContext.print_fps = True

    app = qt.QApplication([])
    window = qt.QWidget()
    vbox = qt.QVBoxLayout()
    window.setLayout(vbox)

    wid = qtgrcontext.QtGrContext()
    vbox.addWidget(wid, 1)

    subwindow = qt.QWidget()
    vbox.addWidget(subwindow, 0)
    hbox = qt.QHBoxLayout()
    subwindow.setLayout(hbox)
    
    button1 = qt.QPushButton("Button 1")
    button1.clicked.connect(lambda : click("button1", wid))
    hbox.addWidget(button1)
    button2 = qt.QPushButton("Button 2")
    button2.clicked.connect(lambda : click("button2", wid))
    hbox.addWidget(button2)

    sys.stderr.write("Showing window.\n")
    window.show()

    mainlooptimer = qtcore.QTimer()
    mainlooptimer.timeout.connect(lambda : mainloop(wid) )
    mainlooptimer.start(1000./30.)
        
    sys.stderr.write("Executing Qt app\n")
    app.exec_()



# ======================================================================

if __name__ == "__main__":
    main()
