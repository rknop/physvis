#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# (c) 2019 by Rob Knop
#
# This file is part of physvis
#
# physvis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# physvis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with physvis.  If not, see <https://www.gnu.org/licenses/>.


# Make documentation with pydoc3 -w ./physvis.py

"""A module for visualizing simple physics constructions in 3D.

Desgined to reimplement a subset of VPython 6 for Python 3, with a standalone display
(i.e. not requiring a web browser for display).  Implemented in pure Python, using
PyOpenGL (specifically, the GL and GLUT modules).

This module is really just a front-end for visual_base.py; this module has the more direct
VPython-like interface.

This module depends on you having the other nonstandard modules installed:

  * numpy
  * scipy
  * PyOpenGL (modules OpenGL.GL and OpenGL.GLUT)

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
down the left mouse button and move the mouse to move the point the display looks at.

The following ojects are available

  — arrow
  — box
  — cylinder
  — cone
  — curve
  — ellipsoid
  — faces
  — helix
  — label
  — ring
  — sphere
  — tetrahedron
  — octahedron
  — icosahedron
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
    up — Rotate the object about its axies in an attempt to make this *local* vector of the object up on the screen
    scale — How much to scale the object (interactis with the amgnitude of axis)
    color — The color of the object (r, g, b) or (r, g, b, a)
    make_trail — True to leave behind a trail
    interval — Only add a trail segment after the object has moved this many times (default: 10)
    retain — Only keep this many trail segments (the most recent ones) (Default: 50)
    context — The display (window or widget) to put this object in.

  The following properties you can't specify when you make the object, but you can change
  later (e.g. with "mycube.x = 0.1):

    x — The x position of the object
    y — The y position of the object
    z — The z position of the object
    sx — The x scale of the object (this is usually the object's "length" where that makes sense)
    sy — The y scale of the object
    sz — The z scale of the object
    visible — Set to False to remove an object from the display, True to put it back in.

  Note that "up" is kind of a strange property.  When you read it, you
  probably don't get anything meaningful.  When you set it, it will try
  to orient the object so that the "up" vector you specified in the
  object's local coordinate system is up on the screen.  If an object is
  created in its default orientation (i.e. without an axis keyword), its
  local coordinate system matches the global coordinate system.  If you
  point the object in anoter direction, but want to have the same side
  of the object upwards, set up to (0, 1, 0).  up only matters right
  when you set it.  If you rotate the object again later, the code will
  not try to keep the same side up; it will just do the most direct
  rotation to get the object oriented in the new direction.  (This
  behavior is probably not the same as what VPython used.)

  Methods:

  These are functions on an object you can call to do things to the object, e.g.

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

(ASIDE: Internally, vector is implemented as a few things on top of a
3-element numpy array, so you can use them wherever you'd use a numpy
array.  However, some things are a bit wonky; for instance, try making
two vectors v1 and v2, and doing (v1*v2).sum().  You'll get back a
vector object with one element, which is broken, and probably means I
didn't subclass properly.  You can fix this by just typecasting the
result to a float.  Really, you're probably better off treating the
vector class as its own thing and not relying too muich on the fact that
it's a numpy array.  For instance, in this example, it's better just to
do v1.dot(v2).)


CONTROLLING DISPLAYS (WINDOWS)

You can get a new display with the function "display()", e.g.:

  disp = display()

That will open a new window; disp is a handle to it.  If you want a
handle to the first display created (which may have been created
automatically when you created a graphic object), the function "scene()"
returns that.

You can get and set various display properties:

  * width — width of the window
  * height — height of the window
  * background — the background color of the display.  Three numbers
                    (rgb) between 0 and 1, where 0 is black.  You can
                    also use color.red, etc.
  * foreground — the default color of newly-created objects.
  * center — (a vector, or a list or tuple of 3 objects) the coordinates
                of the point the camera is looking at.  If the user
                rotates the display, this point will stay centered.
                Default: (0, 0, 0)
  * forward — the direction the camera looks.  The camera will always look
                at the "center" position.  If you set forward, it will move
                the camera to the right place so that looking in the forward
                direction it's looking at center.
                Default: (0, 0, -1)
  * up — A vector in world that will (as best possible) be up on the screen
  * fov — the field of view of the camera.  Default: π/3

  * range — the range of the dispay in world coordinates. For
              historical reasons, it's a three-element vector, but all
              three values have to be the same.  You can set it with a
              single value.  An object of radius range at center will
              have angular size fov (i.e. it will just fill the screen).
              Change this to zoom the display.
              Default: (3, 3, 3)

  * scale — 1/range.  If you set this, it changes range, and vice versa.
              Bigger scale means smaller objects on the screen.

Many of these properties will be overridden if the user uses the mouse.
In particular, whenever the user adjusts the display rotation with the
mouse, "up" will be adjusted so that the +y-axis is (as best possible)
up on the screen.

You can make objects go into the a display by adding "context=disp"
when you make the object, where disp is the value returned from a
display() function call.  Alternatively, if you call "disp.select()",
that will make that display the default for newly created objects.  If
you want to get to the first (probably automatically created) display,
the function "scene()" returns a handle to that first display, so you
can call ".select()" on it and pass it to objects to put them in that
first display.

Currently, all displays are GLUT windows.  In the future, I'm hoping
that you will be able to get a display as a Qt OpenGL widget


DIFFERENCES FROM VPYTHON 6           

Many objects are not implemented. physvis includes some objects not in VPython 6
(e.g. *axis).  (There are also some in visual_base.py, e.g. Icosahedron.)
There may be some parameters some objects take which weren't there in VPython.

An incomplete list things not implemented:

Objects missing:
  * extrusion
  * local lights
  * points
  * pyramid
  * text

Object properties missing or not working:
  * opacity   (Doing this at all well is quite hard.  The variable is there, but ignored)
  * materials (Maybe someday)
  * composite objects with frame
  * You can't delete displays (and object cleanup in general is lacking)
  * You can't append faces to a faces object
  * You can't give an array of colors to a faces object; it's all one color

Global features missing:
  * Automatic zoom updating to make all objects visible
  * custom lighting
  * Widgets, embedding display in a UI library (see below re: wxPython)
  * Graphs (I'll probably never do this; just use matplotlib)
  * Custom mouse and keyboard events
  * Shapes library
  * Paths library

Things Changed:
  * There is no "scene" variable; the "scene()" function gets you the default display.
  * I believe the default range of a display is different (bigger)
  * For label, sizes (height, xoffset, yoffset) are not in the same units as VPython 6
       There's also no reference line, just the (maybe boxed) text.
  * axes() object is new
  * icosahedron() object is new
  * tetrahedron() object is new
  * octahedron() object is new
  * context= parameter of objects is new (I think)
  * For faces, you don't have to specify the normals; it will default to
    flat faces and calculate the normals if you don't give them.
  * You can give a faces (3*n*3) array for n triangles, or a (3*n, 3) array.

wxPython interaction is not implemented; for a very long time, it looked
like wxPython was dead and would not support Python 3.  That's no longer
true, but PyQt (or PySide2, or, really, probably QtPy as a front-end to
both) may still be a better choice.  For now, it uses GLUT to open the
one default window for display.  In the future, I hope to integrate it
with some standard Python widget library.

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

def curve(pos=None, points=None, *args, **kwargs):
    """A curve.

    pos or points — A n×3 array of points that make up the centerline of the curve
    raidus — the radius of the cross-section of the curve
    """
    if pos is None:
        return vb.Curve(points=points, *args, **kwargs)
    else:
        return vb.Curve(points=pos, *args, **kwargs)

def ellipsoid(*args, **kwargs):
    """An ellipsoidal solid.

    subdivisions — higher = more faces (default: 2, which is probably always good)
    length — diameter along x-axis (for unrotated object)
    width — diameter along z-axis (for unrotated object)
    height — diameter along y-axis (for unrotated object)
    """
    return vb.Ellipsoid(*args, **kwargs)

def faces(vertices, normals=None, *args, **kwargs):
    """An arbitrary set of triangles.

    vertices — Either a 3*3*n numpy array or a (3*n, 3) numpy array
               specifing the vertices of all the triangles.  Each
               triangle has 3 vertices, and each vetex has 3 values (x,
               y, z).  Make sure to orient the vertices of each triangle
               so that if you curl the fingers along the direction of
               the three vertices, your thumb points along the outward
               normal of that triangle.
    normals — Either a 3*3*n numpy array, a (3*n, 3) numpy array, or
              None (the default).  If you don't specify this or pass None, the
              code will generate normals that makes each face flat.
    smooth — If True, makes a smooth object (normals at each vertex
             averaged over the adjancet faces).  If False (default),
             faces are flat.  (NOTE: smooth isn't implemented. yet.)
    """

    if not isinstance(vertices, numpy.ndarray):
        vertices = numpy.array(vertices)
    if len(vertices.shape) == 2:
        if vertices.shape[1] != 3 or vertices.shape[0] % 3 != 0:
            raise Exception("faces needs an 3n×3 array if you pass it a 2d array, where n is the number of triangles")
    elif len(vertices.shape) == 1:
        if vertices.shape[0] % 9 != 0:
            raise Exception("faces needs 3n vertices (so 9*n values)")
        vertices = vertices.copy()
        vertices.shape = (vertices.shape[0]//3, 3)
    else:
        raise Exception("faces requires a 1d or 2d array.")

    if normals is not None:
        if not isinstance(normals, numpy.ndarray):
            normals = numpy.array(normals)
        if len(normals.shape) == 2:
            if normals.shape[1] != 3 or normals.shape[0] % 3 != 0:
                raise Exception("faces (normals) needs an 3n×3 array if you pass it a 2d array, where n is the number of triangles")
        elif len(normals.shape) == 1:
            if normals.shape[0] % 9 != 0:
                raise Exception("faces (normals) needs 3n vertices (so 9*n values)")
            normals = normals.copy()
            normals.shape = (normals.shape[0]//3, 3)
        else:
            raise Exception("faces (normals) requires a 1d or 2d array.")
        if normals.shape != vertices.shape:
            raise Exception("faces: if you give normals, must have the same number as you have vertices3")
        
    return vb.Faces(vertices, normals, *args, **kwargs)

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

def ring(*args, **kwargs):
    """A torus.

    radius — The major radius of the torus (the radius to a zero-thickness ring) (default: 0.5)
    thickness — The radius of the walls of the ring (default: 1/5 radius)
    num_circ_points — The number of points along the circle for the ring (def: 36)
    """
    return vb.Ring(*args, **kwargs)

def sphere(*args,**kwargs):
    """A sphere, modelled by default as a 2x subdivided icosahedron (320 faces).

    radius — radius of the sphere (default 1)
    subdivisions — higher = more faces (default: 2) (More than 3 is excessive, even 3 is probably excessive.)
    flat — Set to true for flat face shading rather than smooth shading (default: False)
    subdivisions — Control how smooth the underlying geometry is.  Default: 2.  0 = icosahedron.  >3 = unreasonable.
    """
    return vb.Sphere(*args, **kwargs)

def tetrahedron(*args, **kwargs):
    """A tetrahedron with a point on the +x axis and points 1 unit (by default) from the origin."""
    return vb.Tetrahedron(*args, **kwargs)

def octahedron(*args, **kwargs):
    """An octahedron with (by default) points at ±1 along each of the axes."""
    return vb.Octahedron(*args, **kwargs)

def icosahedron(flat=True, *args, **kwargs):
    """An icosahedron, possibly subdivided.

    radius — The radius of the object (to the vertices) (default: 1)
    flat — If True, render flat faces rather than smooth faces (default: True)
           This property can't be set after object creation.
    subidvisions — How many times to subdivide the icosahedron; each subdivision
                   increases the number of faces 4×.  This property can't be set
                   after object creation.
    """
    return vb.Icosahedron(flat=flat, *args, **kwargs)

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


def label(*args, **kwargs):
    """A label that always faces forward and is the same size regardless of zoom.

    By default, the label is ~1/7 the height of the screen and is an
    italic serif font.  It's positioned so that the center of the bottom
    of the text is at the pos you give it.

    You can give it the following properties at initialization, and you can set them:

    text — the text to render
    font — one of 'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'
    height — The height of the font -- really, the point size of the font (default: 12)
    italic — True or False to italicize (default: True)
    bold — True or False for bold text (default; False)

    pos — a 3d vector, the reference position for where the label is
    color — three values, the r, g, and b values of the text (between 0 and 1)

    units — units for xoffset, yoffset, refheight.  There are three posibilities.
             "display" means very roughly in units of the dispaly size
                -- so 0.5 is approximately half the screen height.  (It's
                not exactly right.)
             "centidisplay" (the default) is hundreths of the display
                (so 25 would be approximately a quarter the screen size)
             "pixels" is not implemented.
    xoffset, yoffset — By default, the label is positioned so that the
           center of the bottom of the label is at the reference point
           specified by pos.  Give these values to offset the label from
           that position; the units are given by the "units" keyword.
    refheight — A "reference height".  The height in units of a
           character in a 12-point font.  Defaults to 15 centidisplay
           units (or 0.15 display units).
    box — Set to True to draw a box around the text (default: True)
    border — Border in points between the text and the box (default: 1)
    linecolor — Color of the border (default: same as text)
    linewidth — Width of the border line in points (default: 0.5)
    visible — (Means the same thing as any other object)

    """
    
    import visual_label
    return visual_label.LabelObject(*args, **kwargs)

# ======================================================================

def scene():

    """Return the first display that was created (perhaps automatically).

    Will be None if you haven't made any displays or graphic objects yet.
    """
    return vb.GrContext._first_context

def display(*args, **kwargs):
    """Create a new display in which you can put objects.

    If you get a display with "disp=display()", you can make that
    display the default for new objects wiht "disp.select()".
    Alternatively, you can explicitly put an object in a given display
    by using the "context=disp" parameter in the functon that creates an
    object.

    See overall module documentation at the top for properties of displays.
    """
    return vb.GrContext.get_new(*args, **kwargs)

# ======================================================================

def rate(fps):
    """Call this e.g. in a while loop in your program to limit the animation speed.

    The while loop will run at most this many times each second.  If the
    computations in the loop take longer than 1/fps seconds, then the
    loop will run slower.
    """
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

# ======================================================================


    
