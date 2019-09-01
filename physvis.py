#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""A module for visualizing simple physics constructions in 3D.

Desgined to reimplement a subset of VPython 6 for Python 3, with a
standalone display (i.e. not requiring a web browser for display).
Implemented in pure Python, using PyOpenGL (GL and GLUT modules).

This module is really just a front-end for visual_base.py; this module
has the more direct VPython-like interface.

Many objects are not implemented. Includes some objects not in VPython 6
(e.g. *axis).  (There are also some visual_base.py,
e.g. FixedLengthCurve and Icosahedron.  There may be some parameters
some objects take which weren't there in VPython.)

An incomplete list things not implemented:

Objects missing:
  * curve (but see visual_base.CylindarStack & visual_base.FixedLengthCurve)
  * extrusion
  * faces
  * label
  * local lights
  * points
  * pyramid
  * ring
  * text

Object properties missing or not working:
  * up        (It's there, but doesn't do anything, and throws errors)
  * opacity   (Doing this at all well is quite hard.  The variable is there, but ignored)
  * materials (Maybe someday)
  * composite objects with frame

Global features missing:
  * More than one display window
  * custom lighting
  * Widgets, embedding display in a UI library (see below re: wxPython)
  * Graphs (I'll probably never do this; just use matplotlib)
  * Custom mouse and keyboard events
  * Shapes library
  * Paths library

wxPython interaction is not implemented; for a very long time, it looked
like wxPython was dead and would not support Python 3.  That's no longer
true, but PyQt may still be a better choice.  For now, it uses GLUT to
open the one default window for display.

"""

import math
import numpy
import visual_base as vb
from visual_base import color
from visual_base import vector

def arrow(*args, **kwargs):
    """An arrow with a square cross-section shaft.

    Oriented along the x-axis by default.

    shaftwidth — The width of the shaft (default: 0.1 * length)
    headwidth — The width of the head (default: 2 * shaftwidth)
    headlength — The length of the head (default: 3 * shaftidth)
    fixedwidth — If False (default), all of the dimensions scale when you change the length of the arrow.
                 If True then shaftwidth, headwidth, and headlength stays fixed.
    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    """
    return vb.Arrow(*args, **kwargs)

def box(*args, **kwargs):
    """A rectangular solid with dimenions (x,y,z) = (length,height,width)

    length — The size of the box along the x-axis (left-right with default camera orientation)
    height — The size of the box along the y-axis (up-down with default camera orientation)
    width — The size of the box along the z-axis (in-out with default camera orientatino)
    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    """
    return vb.Box(*args, **kwargs)

def cylinder(*args, **kwargs):
    """A cylinder, oriented by default along the x-axis.

    radius — The transverse radius of the cylinder.
    num_edge_points — How many faces the end polygons have (default: 16)
    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    """
    return vb.Cylinder(*args, **kwargs)

def cone(*args, **kwargs):
    """A cone with its base at the origin and its tip at 1,0,0

    radius — the radius of the circle that makes the base of the cone.  (Really a 12-sided polygon, smooth shaded.)
    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    """
    return vb.Cone(*args, **kwargs)

def ellipsoid(*args, **kwargs):
    """An ellipsoidal solid.

    subdivisions — higher = more faces (default: 2, which is probably always good)
    length — diameter along x-axis (for unrotated object)
    width — diameter along z-axis (for unrotated object)
    height — diameter along y-axis (for unrotated object)
    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    """
    return vb.Ellipsoid(*args, **kwargs)

def helix(*args, **kwargs):
    """A helix (spring), rendered as a tube around a helical path.

    Initially oriented along the x-axis, with the first point at +z, and
    winding left-handed.  The object's position is at one end of the
    center line of the spring.

    radius — The radius of the whole spring
    coils — The number of coils in the spring (can be fractional, but you'll get an approximation)
    length — The length of the spring (redundant with the length of axis)
    num_circ_points — The number of points on the path in one winding of the spring (default: 12)
    thickness — The thickness of the actual spring wire (default: 0.05 * radius)
    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    """
    return vb.Helix(*args, **kwargs)

def sphere(*args,**kwargs):
    """A sphere, modelled by default as a 2x subdivided icosahedron (320 faces).

    radius — radius of the sphere (default 1)
    subdivisions — higher = more faces (default: 2) (More than 3 is excessive, even 3 is probably excessive.)
    flat — Set to true for flat face shading rather than smooth shading (default: False)
    subdivisions — Control how smooth the underlying geometry is.  Default: 2.  0 = icosahedron.  >3 = unreasonable.
    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    """
    return vb.Sphere(*args, **kwargs)


def xaxis(*args, **kwargs):
    """A red arrow at the origin of length 1 pointing in the x-direction"""
    return vb.Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(1, 0, 0), color=color.red)

def yaxis(*args, **kwargs):
    """A green arrow at the origin of length 1 pointing in the y-direction"""
    return vb.Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(0, 1, 0), color=color.green)

def zaxis(*args, **kwargs):
    """A blue arrow at the origin of length 1 pointing in the z-direction"""
    return vb.Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(0, 0, 1), color=color.blue)

def axes(*args, **kwargs):
    """Returns a tuple of three objects, an x-axis, a y-axis, and a z-axis.  Call this to quickly generate axes."""
    return ( xaxis(), yaxis(), zaxis() )

# ======================================================================

def rate(fps):
    """Call this e.g. in a while loop in your program to limit the animation speed.

    The while loop will run at most this many times each second.  If the
    computations in the loop take longer than 1/fps seconds, then the
    animation will run slower."""
    vb.rate(fps)

def radians(deg):
    """Pass degrees, get back radians."""
    return 2.*math.pi * deg / 360.

def degrees(rad):
    """Pass radians, get back degrees."""
    return 360. * rad / (2.*math.pi)

# ======================================================================

def norm(A):
    """Return a unit vector along the passed vector."""
    if type(A) is vector:
        return A.norm()
    else:
        return vector(A).norm()

def mag(A):
    """Return the magnitude of the passed vector."""
    return math.sqrt(numpy.square(A).sum())

def mag2(A):
    """Return the magnitude squared of the passed vector.

    This is more efficient than doing mag(A)**2
    """
    return numpy.square(A).sum()

def cross(A, B):
    """Return the cross product of two vectors."""
    if type(A) is vector:
        return A.cross(B)
    else:
        return vector(A).cross(B)

def proj(A, B):
    """Return the projection of the first vector along the second.

    proj(A, B) = dot(A, norm(B)) * norm(B)
    """
    if type(A) is vector:
        return A.proj(B)
    else:
        return vector(A).proj(B)

def comp(A, B):
    """Return the component of the first vector along the second (A·B / |B|)"""
    if type(A) is vector:
        return A.comp(B)
    else:
        return vector(A).comp(B)

def diff_angle(A, B):
    """Return the angle between two vectors (radians)."""
    if type(A) is vector:
        return A.diff_angle(B)
    else:
        return vector(A).diff_angle(B)

def rotate(A, theta, B):
    """Return vector A rotated by theta around vector B."""
    if type(A) is vector:
        return A.rotate(theta, B)
    else:
        return vector(A).rotate(theta, B)

def astuple(A):
    """Return a 3-element tuple with the components of a vector."""
    if type(A) is vector:
        return A.astuple()
    else:
        return tuple(A)
