#/usr/bin/python3
# -*- coding: utf-8 -*-

from physvis import *

def main():
    axeses = axes()
    xlab = label(pos=(1., 0., 0.), color=color.red, text="x", border=True,
                 refheight=15, xoffset=7.5, yoffset=0.)
    ylab = label(pos=(0., 1., 0.), color=color.green, text="y", border=True,
                 refheight=15, xoffset=0., yoffset=0.)
    zlab = label(pos=(0., 0., 1.), color=color.blue, text="z", border=True,
                 refheight=15, xoffset=-7.5, yoffset=0.)

    countdownx = 60
    countdowny = 45
    
    while True:
        rate(30)
        countdownx -= 1
        if countdownx <= 0:
            xlab.visible = not xlab.visible
            countdownx = 60
        countdowny -= 1
        if countdowny <= 0:
            ylab.visible = not ylab.visible
            countdowny = 45
            

# ======================================================================

if __name__ == "__main__":
    main()
