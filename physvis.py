#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
import numpy
import visual_base as vb
from visual_base import color
from visual_base import vector

def arrow(*args, **kwargs):
    return vb.Arrow(*args, **kwargs)

def box(*args, **kwargs):
    return vb.Box(*args, **kwargs)

def cylinder(*args, **kwargs):
    return vb.Cylinder(*args, **kwargs)

def cone(*args, **kwargs):
    return vb.Cone(*args, **kwargs)

def ellipsoid(*args, **kwargs):
    return vb.Ellipsoid(*args, **kwargs)

def helix(*args, **kwargs):
    return vb.Helix(*args, **kwargs)

def sphere(*args,**kwargs):
    return vb.Sphere(*args, **kwargs)


def xaxis(*args, **kwargs):
    return vb.Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(1, 0, 0), color=color.red)

def yaxis(*args, **kwargs):
    return vb.Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(0, 1, 0), color=color.green)

def zaxis(*args, **kwargs):
    return vb.Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(0, 0, 1), color=color.blue)

def axes(*args, **kwargs):
    return ( xaxis(), yaxis(), zaxis() )

# ======================================================================

def rate(fps):
    vb.rate(fps)

def radians(deg):
    return 2.*math.pi * deg / 360.

def degrees(rad):
    return 360. * rad / (2.*math.pi)

# ======================================================================

def norm(A):
    if type(A) is vector:
        return A.norm()
    else:
        return vector(A).norm()

def mag(A):
    return math.sqrt(numpy.square(A).sum())

def mag2(A):
    return numpy.square(A).sum()

def norm(A):
    if type(A) is vector:
        return A.norm()
    else:
        return vector(A).norm()

def cross(A, B):
    if type(A) is vector:
        return A.cross(B)
    else:
        return vector(A).cross(B)

def proj(A, B):
    if type(A) is vector:
        return A.proj(B)
    else:
        return vector(A).cross(B)

def comp(A, B):
    if type(A) is vector:
        return A.comp(B)
    else:
        return vector(A).comp(B)

def diff_angle(A, B):
    if type(A) is vector:
        return A.diff_angle(B)
    else:
        return vector(A).diff_angle(B)

def rotate(A, theta, B):
    if type(A) is vector:
        return A.rotate(theta, B)
    else:
        return vector(A).rotate(theta, B)

def astuple(A):
    if type(A) is vector:
        return A.astuple()
    else:
        return tuple(A)

