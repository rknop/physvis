#/usr/bin/python3
# -*- coding: utf-8 -*-

from physvis import *

def main():
    axeses = axes()
    xlab = label(pos=(1., 0., 0.), color=color.red, units='centidisplay', xoffset=50, yoffset=0., text="x", border=True)
    ylab = label(pos=(0., 1., 0.), color=color.green, xoffset=0., yoffset=0., text="y", border=True)
    zlab = label(pos=(0., 0., 1.), color=color.blue, xoffset=0., yoffset=0., text="z", border=True)

    while True:
        rate(30)

# ======================================================================

if __name__ == "__main__":
    main()
