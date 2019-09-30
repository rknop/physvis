#!/usr/bin/python3
# -*- coding: utf-8 -*-

import physvis
import random

def main():
    spheres = []
    for i in range(2000):
        spheres.append( physvis.sphere( pos = ( random.random() * 5 - 2.5,
                                                random.random() * 5 - 2.5,
                                                random.random() * 5 - 2.5 ),
                                        radius=0.05,
                                        color = ( random.random(),
                                                  random.random(),
                                                  random.random() ) ) )

    while True:
        physvis.rate(30)


# ======================================================================

if __name__ == "__main__":
    main()
