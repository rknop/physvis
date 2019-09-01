#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""A module for visualizing simple physics constructions in 3D.

Desgined to reimplement a subset of VPython 6 for Python 3, with a standalone display
(i.e. not requiring a web browser for display).  Implemented in pure Python, using
PyOpenGL (specifically, the GL and GLUT modules).

This module is really just a front-end for visual_base.py; this module has the more direct
VPython-like interface.

USING OBJECTS:

You must be running python3.  physvis is not compatible with python2.

See "bouncing_ball.py" or "rotating_spring.py" as an example.

At the top of your code, include:

  from physvis import *

(If you know what you're doing, you may prefer something like "import physvis as vis".)

Create objects just by calling the appropriate function.  For instance, if you write in your code:

   mycube = box()

it will open up a window with a 3d cube that is 1 unit on a side, colored white.  If later
you change an object's parameters, e.g.:

   mycube.color = color.red

the display will update accordingly.  See example "rotating_springs.py" for an example.
(See "vibrating_array.py" for a more complicated example that uses scipy to solve the
equations of motion.)

In your code, you will need to have a loop (e.g. a "while True:" loop) that does whatever
you want to do (updating object positions, colors, doing other calculations).  Inside this
loop, put the statement

  rate(30)

This does two things.  First, it allows the display to update.  Second, it limits your
loop to run no more than 30 times a second.  (You can replace 30 with a different number,
of course.  It doesn't make sense to do more than 30 or 60 frames per second.  If you want
to do calculations more often than that, do multiple calculations in your loop, and only
update object properties once per loop.)  Your loop may run slower than that if the
computer takes more than 1/30 seconds to process everything in your loop.

The user can rotate the orientation of the display by holding down the right mouse button
and moving the mouse.  Roll the mouse wheel, or hold down the middle mouse button and move
the mouse to zoom the display (really, just move the camera forwards and backwards).  Hold
down the left mouse button and move the mouse to move the display in a slightly
complicated way.  (If you haven't rotated the camera, then the plane of the screen is the
x-y plane, and it moves the display in an obvious way.  If you have, really it moves the
whole world in the x-y plane.)

The following ojects are available

  — arrow
  — box
  — cylinder
  — cone
  — ellipsoid
  — helix
  — sphere
  — xaxis
  — yaxis
  — zaxis
  — axes  (actually returns a tuple of 3 objects, not just one object)


OBJECT PROPERTIES AND METHODS:

Objects (box, cylinder, etc) all (or at least most) have the following properties and
methods.  Each individual type of object will have additional properties you can specify
when creating the object, and that you can change later.  See the documentation below on
each object for more information.

  Properties.  You can set these when you create the object, e.g.:

        mycube = box( pos=(0, 1, 0), color=color.red )

  or you can set them later by changing the property, e.g.:

        mycube.pos = (0, 1.1, 0)

    pos — The position of the object (vector)
    axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
    up — not implemented
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)

  The following properties you can't specify when you make the object, but you can change
  later (e.g. with "mycube.x = 0.1):

    x — The x position of the object
    y — The y position of the object
    z — The z position of the object
    sx — The x scale of the object (this is usually the object's "length" where that makes sense)
    sy — The y scale of the object
    sz — The z scale of the object
    visible — Set to False to remove an object from the display, True to put it back in

  Methods.  These are functions on an object you can call to do things to the object,, e.g.

     mycube.rotate( 0.785, (0, 1, 0) )

  to rotate the object mycube by ~45⁰ (0.785,rougly π/4, radians) about an axis pointing
  in the +y direction and going through the object's origin.

    rotate(angle, axis, origin) — Rotate object by angle about an axis that goes through
                                  origin and is oriented along the parameter axis.  A
                                  right-handed rotation.  If you don't specify axis, it
                                  rotates around the object's own axis (it's "local"
                                  x-axis); if you dont' specify origin, it rotates about
                                  the objects origin.  (For boxes and spheres, the
                                  object's origin is the center; for most other objects,
                                  it's the center of the object's base.

VECTOR OBJECTS

For convenience, there's also a vector() class to represent a three-dimensional vector.
You can make a vector with, for example:

   v = vector( (1, 0.5, 0) )

The "pos" property of physvis objects are of this vector type.

Many standard operations work with vectors.  You can add and subtract them.  You can
multiply or divide them by a scalar.  Plus, vectors have the following properties and methods:

  v.mag — returns the magnitude of vector v
  v.mag2 — returns the magnitude squared of vector v
  v.norm — returns a unit vector in the same direction as vector v
  v.cross(u) — returns the cross product v × u
  v.dot(u) — returns the dot product v · u
  v.proj(u) — returns v projected along u, or ( v·u / |u| ) u
  v.comp(u) — returns the component of v along u, or ( v·u / |u| )
  v.diff_angle(u) — returns the angle between v and u (in radians)
  v.rotate(theta, u) — returns a new vector rotated by angle theta about u

There are also functions corresponding to each of the properties and methods above, e.g.

  cross(v, u)

returns v.cross(u).

(ASIDE: Internally, vector is implemented as a few things on top of a 3-element numpy
array, so you can use them wherever you'd use a numpy array.  However, some things are a
bit wonky; for instance, try making two vectors v1 and v2, and doing (v1*v2).sum().
You'll get back a vector object with one element, which is broken, and probably means I
didn't subclass properly.  You can fix this by just typecasting the result to a float.
Really, you're probably better off treating the vector class as its own thing and not
relying too muich on the fact that it's a numpy array.)


DIFFERENCES FROM VPYTHON 6           

Many objects are not implemented. physvis includes some objects not in VPython 6
(e.g. *axis).  (There are also some visual_base.py, e.g. FixedLengthCurve and Icosahedron.
There may be some parameters some objects take which weren't there in VPython.)

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
  * Automatic zoom updating to make all objects visible
  * custom lighting
  * Widgets, embedding display in a UI library (see below re: wxPython)
  * Graphs (I'll probably never do this; just use matplotlib)
  * Custom mouse and keyboard events
  * Shapes library
  * Paths library

wxPython interaction is not implemented; for a very long time, it looked like wxPython was
dead and would not support Python 3.  That's no longer true, but PyQt may still be a
better choice.  For now, it uses GLUT to open the one default window for display.  In the
future, I hope to integrate it with some standard Python widget library.

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
    """
    return vb.Arrow(*args, **kwargs)

def box(*args, **kwargs):
    """A rectangular solid with dimenions (x,y,z) = (length,height,width)

    length — The size of the box along the x-axis (left-right with default camera orientation)
    height — The size of the box along the y-axis (up-down with default camera orientation)
    width — The size of the box along the z-axis (in-out with default camera orientatino)
    """
    return vb.Box(*args, **kwargs)

def cylinder(*args, **kwargs):
    """A cylinder, oriented by default along the x-axis.

    radius — The transverse radius of the cylinder.
    num_edge_points — How many faces the end polygons have (default: 16)
    """
    return vb.Cylinder(*args, **kwargs)

def cone(*args, **kwargs):
    """A cone with its base at the origin and its tip at 1,0,0

    radius — the radius of the circle that makes the base of the cone.  (Really a 12-sided polygon, smooth shaded.)
    """
    return vb.Cone(*args, **kwargs)

def ellipsoid(*args, **kwargs):
    """An ellipsoidal solid.

    subdivisions — higher = more faces (default: 2, which is probably always good)
    length — diameter along x-axis (for unrotated object)
    width — diameter along z-axis (for unrotated object)
    height — diameter along y-axis (for unrotated object)
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
    """
    return vb.Helix(*args, **kwargs)

def sphere(*args,**kwargs):
    """A sphere, modelled by default as a 2x subdivided icosahedron (320 faces).

    radius — radius of the sphere (default 1)
    subdivisions — higher = more faces (default: 2) (More than 3 is excessive, even 3 is probably excessive.)
    flat — Set to true for flat face shading rather than smooth shading (default: False)
    subdivisions — Control how smooth the underlying geometry is.  Default: 2.  0 = icosahedron.  >3 = unreasonable.
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

def dot(A, B):
    """Return the dot product of the two vectors."""
    return numpy.dot(A, B)

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
