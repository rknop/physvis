#!/usr/bin/python3
# -*- coding: utf-8 -*-

import visual_base as vb

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

def rate(fps):
    vb.rate(fps)

