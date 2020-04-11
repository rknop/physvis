#/usr/bin/python3
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

"""visual_base is a basic framework and scene graph for easily creating,
displaying, and updating simple 3d objects useful for visualization of physics.
"""

import sys
import math
import time
import queue
import threading
# import multiprocessing
import random
import ctypes
import itertools

import numpy
import numpy.linalg

from quaternions import *
from rater import Rater
# from physvis_observer import Subject, Observer    # comes in grcontext
from grcontext import *
from object_collection import *

_first_context = None

# ======================================================================
# rate()

def rate(fps):
    """Call this in the main loop of your program to have it run at most every 1/fps seconds."""
    rater = Rater.get()
    rater.rate(fps)

def exit_whole_program():
    """Call this to have the program quit next time you call rate()."""
    Rater.exit_whole_program()
    

# ======================================================================
# A special case 3-element numpy.ndarray.  Can't create them as
#   views or slices of other things.  This might not be exactly what
#   I want; not 100% sure.  I need to experiment with old VPython to
#   find out.

class vector(numpy.ndarray):
    """A 3-element vector of floats that represents a physical vector quantity (displacement, velocity, etc.).

    Create a new vector with just:

          v = vector()

    (which initializes it to (0, 0, 0)), or with:

          v = vector( (0, 1, 0) )

    The argument you pass to vector() must be a sequence, i.e. a tuple,
    list, numpy array, vector, or something else that has three
    elements.

    """
    
    def __new__(subtype, vals=(0., 0., 0.), copyvector=None):
        if copyvector is not None:
            return super().__new__(subtype, 3, float, copyvector)
        if len(vals) != 3:
            err = "Need 3 values to initialize a vector, got {}\n".format(len(vals))
            raise IndexError(err)
        # I *hope* that when I call  __new__, it doesn't copy the data again.
        #   I don't think it does.
        tmp = numpy.array(vals, dtype=float)
        return super().__new__(subtype, (3), float, tmp)

    @property
    def mag(self):
        """The magnitude of the vector."""
        return math.sqrt(numpy.square(self).sum())

    @mag.setter
    def mag(self, val):
        self /= self.mag
        self *= val

    @property
    def mag2(self):
        """The square of the magnitude of the vector.

        Faster than getting vector.mag and then squaring.
        """
        return numpy.square(self).sum()
        
    def norm(self):
        """Returns a new vector that is the unit vector in the same direction."""
        
        vec = vector(self)/self.mag
        return vec

    def cross(self, B, **unused_kwargs):
        """Returns the cross product of this vector and B."""
        
        if type(B) is not vector:
            B = vector(B)
        return vector(copyvector = numpy.cross(self, B))

    def proj(self, B, **unused_kwargs):
        """Returns this vector projected on to B.

        A.proj(B) = A.dot(B.norm()) * B.norm()
        """
        
        if type(B) is not vector:
            B = vector(B)
        Bn = B.norm()
        return vector(copyvector = Bn * self.dot(Bn))

    def comp(self, B, **unused_kwargs):
        """Returns the component of this vector along B.

        A.comp(B) = A.dot(B.norm())
        """

        if type(B) is not vector:
            B = vector(B)
        return self.dot(B.norm())

    def diff_angle(self, B, **unused_kwargs):
        """Returns the angle between this vector and V."""
        
        if type(B) is not vector:
            B = vector(B)
        return math.acos( self.dot(B) / (self.mag * B.mag) )

    def rotate(self, theta, B, **unused_kwargs):
        """Rotates this vector by angle theta about vecytor B."""
        
        if type(B) is not vector:
            B = vector(B)
        B = B.norm()
        st = math.sin(theta/2.)
        ct = math.cos(theta/2.)
        roted = quaternion_rotate(self, numpy.array( [st*B[0], st*B[1], st*B[2], ct] ))
        return vector(copyvector = roted)

    # I don't know if this is really faster than astype(self)
    def astuple(self):
        """Return a 3-element tuple of the vector components."""
        
        return ( self[0], self[1], self[2] )
    
    
# ======================================================================

class color(object):
    """A helper class for creating colors (which are just 3-element numpy arrays.

    Defined colors:
    color.red — [1, 0, 0]
    color.green — [0, 1, 0]
    color.blue — [0, 0, 1]
    color.yellow — [1, 1, 0]
    color.cyan — [0, 1, 1]
    color.magenta — [1, 0, 1]
    color.orange — [1, 0.5, 0]
    color.black — [0, 0, 0]
    color.white — [1, 1, 1]
    """
    
    red = numpy.array( [1., 0., 0.] )
    green = numpy.array( [0., 1., 0.] )
    blue = numpy.array( [0., 0., 1.] )
    yellow = numpy.array( [1., 1., 0.] )
    cyan = numpy.array( [0., 1., 1.] )
    magenta = numpy.array( [1., 0., 1.] )
    orange = numpy.array( [1., 0.5, 0.] )
    black = numpy.array( [0., 0. ,0.] )
    white = numpy.array( [1., 1., 1.] )

    def gray(val):
        """Returns a grey color; val=0 is black, val=1 is white."""
        return numpy.array( [val, val, val] )

    def grey(val):
        """Returns a grey color; val=0 is black, val=1 is white."""
        return gray(val)
    
# ======================================================================
# ======================================================================
# ======================================================================

class GrObject(Subject):
    """Base class for all graphical objects (Box, Sphere, etc.)"""
    
    def __init__(self, context=None, pos=None, axis=None, up=None, scale=None,
                 color=None, opacity=None, make_trail=False, interval=10, retain=50,
                 trail_radius = 0.02, trail_color=None, *args, **kwargs):
        """Parameters:
        
        context — the context in which this object will exist
        pos — The position of the object (vector)
        axis — The orientation of the object, and, if it's not normalized, the scale along its standard axis
        up — not implemented
        scale — How much to scale the object (interactis with the amgnitude of axis)
        color — The color of the object (r, g, b) or (r, g, b, a)
        make_trail — True to leave behind a trail
        interval — Only add a trail segment after the object has moved this many times (default: 10)
        retain — Only keep this many trail segments (the most recent ones) (Default: 50)
        trail_radius — radius of trail cross-section (def: 0.02)
        trail_color — color of trail (def: same as object)
        """

        super().__init__(*args, **kwargs)

        self._object_type = GLObjectCollection._OBJ_TYPE_SIMPLE
        self._make_trail = False
        self._trail = None
        self.num_triangles = 0
        self._visible = True

        # sys.stderr.write("Starting GrObject.__init__")

        if context is None:
            self.context = GrContext.get_default_instance()
        else:
            self.context = context

        self.draw_as_lines = False

        self._rotation = numpy.array( [0., 0., 0., 1.] )    # Identity quaternion

        if pos is None:
            self._pos = vector([0., 0., 0.])
        else:
            self._pos = vector(pos)

        if scale is None:
            self._scale = numpy.array([1., 1., 1.])
        else:
            self._scale = scale

        self.colordata = None

        if color is None and opacity is None:
            self._color = numpy.array( self.context.default_color, dtype=numpy.float32 )
        elif color is None:
            self._color = numpy.empty(4, dtype=numpy.float32)
            self._color[0:3] = context.default_color[0:3]
            self._color[3] = opacity
        else:
            self._color = numpy.empty(4, dtype=numpy.float32)
            self._color[0:3] = numpy.array(color)[0:3]
            if opacity is None:
                self._color[3] = 1.
            else:
                self._color[3] = opacity

        # ROB, write an interface for these
        self._specular_strength = 0.75
        self._specular_pow = 32
                
        self.model_matrix = numpy.array( [ [ 1., 0., 0., 0. ],
                                           [ 0., 1., 0., 0. ],
                                           [ 0., 0., 1., 0. ],
                                           [ 0., 0., 0., 1. ] ], dtype=numpy.float32)
        self.inverse_model_matrix = numpy.array( [ [ 1., 0., 0., 0.],
                                                   [ 0., 1., 0., 0.],
                                                   [ 0., 0., 1., 0.] ], dtype=numpy.float32)

        self.vertexdata = None
        self.normaldata = None
        self.matrixdata = None
        self.normalmatrixdata = None

        self._axis = vector([1., 0., 0.])
        self._up = vector([0., 1., 0.])

        if axis is not None:
            self.axis = vector(axis)

        if up is not None:
            self.up = numpy.array(up)

        self._interval = interval
        self._nexttrail = interval
        self._retain = retain
        self._trail_radius = trail_radius
        self._trail_color = trail_color
        self.make_trail = make_trail

    def finish_init(self):
        self.update_model_matrix()
        self.context.add_object(self)

    @property    
    def pos(self):
        """The position of the object (a vector).  Set this to move the object."""
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = vector(value)
        self.update_model_matrix()
        self.update_trail()

    @property
    def x(self):
        """The x-component of object position."""
        return self._pos[0]

    @x.setter
    def x(self, value):
        self._pos[0] = value
        self.update_model_matrix()
        self.update_trail()

    @property
    def y(self):
        """The y-component of object position."""
        return self._pos[1]

    @y.setter
    def y(self, value):
        self._pos[1] = value
        self.update_model_matrix()
        self.update_trail()

    @property
    def z(self):
        """The z-component of object position."""
        return self._pos[2]

    @z.setter
    def z(self, value):
        self._pos[2] = value
        self.update_model_matrix()
        self.update_trail()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, scale must have 3 elements.")
            sys.exit(20)
        self._scale = numpy.array(value)
        self.update_model_matrix()

    @property
    def sx(self):
        return self._scale[0]

    @sx.setter
    def sx(self, value):
        self._scale[0] = value
        self.update_model_matrix()

    @property
    def sy(self):
        return self._scale[1]

    @sy.setter
    def sy(self, value):
        self._scale[1] = value
        self.update_model_matrix()

    @property
    def sz(self):
        return self._scale[2]

    @sz.setter
    def sz(self, value):
        self._scale[2] = value
        self.update_model_matrix()

    @property
    def axis(self):
        """The orientation of the object.  Set this to rotate (and maybe stretch) the object.

        Objects by default have an axis along the x-axis, so if you pass
        [1,0,0], you'll get the default object orientation.  The
        magnitude of axis scales the object along it's primary axis.
        (The meaning of the primary axis depends on the type of object.)

        """
        
        return self._axis

    @axis.setter
    def axis(self, value):
        if len(value) != 3:
            raise Exception("axis must have 3 values")
        newaxis = numpy.array( value, dtype=float )
        axismag = math.sqrt( newaxis[0]*newaxis[0] + newaxis[1]*newaxis[1] + newaxis[2]*newaxis[2] )
        if axismag < 1e-8:
            raise Exception("axis too short")
        newaxis /= axismag

        # # This code will figure out the orientation from scratch.
        # # The problem is that it doesn't lead to smooth rotations.
        # self._axis = newaxis
        # self._scale[0] = axismag
        # self.set_object_rotation()

        # Figure out the direct rotation to go from self._axis to
        #   newaxis, and add that on top of current rotation.  (I'm
        #   a little worried about accumulating precision errors.)
        #   Rotate about self._axis × newaxis (normalized)

        # The dot product self._axis · newaxis gives the cos of the angle to rotate (both vectors are normalized)
        # But, because of floating point inefficiencies, I still gotta clip it
        cosrot = self._axis[0]*newaxis[0] + self._axis[1]*newaxis[1] + self._axis[2]*newaxis[2]
        if cosrot > 1.: cosrot = 1.
        elif cosrot < -1.: cosrot = -1.

        # If the new axis is parallel or antiparallel, then we can't use
        #   axis cross newaxis as the rotation axis
        if 1-math.fabs(cosrot) < 1e-8:
            if cosrot > 0.:
                # No actual rotation (well, dinky)
                if self._scale[0] != axismag:
                    self._scale[0] = axismag
                    self.update_model_matrix()
                return
            else:
                # newaxis is opposite self._axis.  Try crossing with zhat to get a rotaxis
                rotax = numpy.array( [ self._axis[1], -self._axis[0], 0. ] )
                # If that didn't work, then use yhat
                rotaxmag = math.sqrt( rotax[0]*rotax[0] + rotax[1]**rotax[1] + rotax[2]*rotax[2] )
                if rotaxmag < 1e-10:
                    rotax = numpy.array( [ -self._axis[2], 0., self._axis[0] ] )
                    rotaxmag = math.sqrt( rotax[0]*rotax[0] + rotax[1]*rotax[1] + rotax[2]*rotax[2] )
                    
        else:
            rotax = numpy.array( [ self._axis[1]*newaxis[2] - self._axis[2]*newaxis[1],
                                   self._axis[2]*newaxis[0] - self._axis[0]*newaxis[2],
                                   self._axis[0]*newaxis[1] - self._axis[1]*newaxis[0] ] )
            rotaxmag = math.sqrt( numpy.square(rotax).sum() )

        rotax /= rotaxmag

        cosrot_2 = math.sqrt( (1+cosrot) / 2. )
        sinrot_2 = math.sqrt( (1-cosrot) / 2. )

        self._rotation = quaternion_multiply( [ sinrot_2 * rotax[0],
                                                sinrot_2 * rotax[1],
                                                sinrot_2 * rotax[2],
                                                cosrot_2 ] , self._rotation )
        self._axis = numpy.array( newaxis )
        self._scale[0] = axismag
        self.update_model_matrix()
            
    @property
    def up(self):
        """A vector in the object's frame tries to be up on the screen.

        If you read this, the results could be meaningless.

        Pass a vector in the object's frame; the object will be rotated
        around its axis in an attempt to make that vector "up" on the
        screen.  Only the y- and z- components of up matter, as the x
        component in the object's frame points along the object's axis.
        """
        return self._up
        
    @up.setter
    def up(self, value):
        if len(value) != 3:
            sys.stderr.write("ERROR, up must have 3 elements.")
        # Only the component in the yz plane matters.  Normalize.
        yzmag = math.sqrt(value[1]**2 + value[2]**2)
        # Punt if this is zero
        if yzmag < 1e-8:
            self._up = numpy.array( [0., 1., 0.] )
        else:
            self._up = numpy.array( [ 0., value[1]/yzmag, value[2]/yzmag ] )
        self.set_object_rotation()

    @property
    def trail_radius(self):
        return self._trail_radius

    @trail_radius.setter
    def trail_radius(self, val):
        self._trail_radius = val
        if self._trail is not None:
            self._trail.radius = self._trail_radius

    @property
    def rotation(self):
        """Returns (for now) a quaternion representing the rotation of the object away from [1, 0, 0]"""
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if len(value) != 4:
            sys.sderr.write("rotation is a quaternion, needs 4 values\n")
            sys.exit(20)
        self._rotation = numpy.array(value)
        self.update_model_matrix()

    def rotate(self, angle, axis=None, origin=None):
        """Rotate the object by angle angle about axis axis."""
        if axis is None:
            axis = self.axis
        axis = numpy.array(axis)
        axis /= math.sqrt(numpy.square(axis).sum())
        s = math.sin(angle/2.)
        c = math.cos(angle/2.)
        q = numpy.array( [axis[0]*s, axis[1]*s, axis[2]*s, c] )
        if origin is not None:
            if len(origin) != 3:
                raise Exception("Error, origin must have 3 values.")
            origin = numpy.array(origin)
            relpos = self._pos - origin
            relpos = quaternion_rotate(relpos, q)
            self.pos = origin + relpos
        self.rotation = quaternion_multiply(q, self.rotation)

    def set_object_rotation(self):
        """Figures out what the quaternion self._rotation should be from self._axis and self._up.

        (Both self._axis and self._up must be normalized
        """
        # θ is the angle off of the x-axis
        # φ is the angle in the y-z plane off of y towards z (i.e. about x)
        # costheta = math.sqrt(1 - self._axis[1])
        try:
            costheta = self._axis[0]
            costheta_2 = math.sqrt( (1+costheta) / 2. )
            sintheta_2 = math.sqrt( (1-costheta) / 2. )
        except ValueError:
            import pdb; pdb.set_trace()
            
        # Make sure we aren't going to divide by (close to) 0
        yzmag = math.sqrt( self._axis[1]**2 + self._axis[2]**2 )
        if yzmag < 1e-12:
            # Object is oriented effectively along ±x
            if self._axis[1] < 0.:
                baserot = numpy.array( [0., 1., 0., 0.] )      # rot by π about y
            else:
                baserot = numpy.array( [0., 0., 0., 1.] )      # no rotatiob
        else:
            cosphi = self._axis[1] / yzmag
            cosphi_2 = math.sqrt( (1+cosphi) / 2. )
            sinphi_2 = math.sqrt( (1-cosphi) / 2. )
            if self._axis[2] < 0.:
                # gotta rotate about -x
                xrot = -1.
            else:
                xrot = 1.
            # This is the quaternion for a rotation of θ about z followed by a rotation of φ about ±x (I HOPE!)
            baserot = numpy.array( [ costheta_2 * sinphi_2 * xrot,
                                    -sintheta_2 * sinphi_2 * xrot, 
                                     sintheta_2 * cosphi_2,
                                     costheta_2 * cosphi_2 ] )
        # Finally, rotate around axis by an angle determined by up
        if self._up[2] < 0.:
            psirot = -1
        else:
            psirot = 1.
        cospsi = self._up[1]
        cospsi_2 = math.sqrt( (1+cospsi)/2. )
        sinpsi_2 = math.sqrt( (1-cospsi)/2. )
        self._rotation = quaternion_multiply( [ sinpsi_2 * self._axis[0] * psirot,
                                                sinpsi_2 * self._axis[1] * psirot,
                                                sinpsi_2 * self._axis[2] * psirot,
                                                cospsi_2 ] , baserot )

        self.update_model_matrix()
        
    def update_model_matrix(self):
        """(Internal function to update stuff needed by OpenGL.)"""
        q = self._rotation
        s = 1./( q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3] )
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        rot = numpy.array(
            [[ 1.-2*s*(q1*q1+q2*q2) ,    2*s*(q0*q1-q2*q3) ,    2*s*(q0*q2+q1*q3)],
             [    2*s*(q0*q1+q2*q3) , 1.-2*s*(q0*q0+q2*q2) ,    2*s*(q1*q2-q0*q3)],
             [    2*s*(q0*q2-q1*q3) ,    2*s*(q1*q2+q0*q3) , 1.-2*s*(q0*q0+q1*q1)]],
            dtype=numpy.float32)
        # Inverse quaternion, just flip the sign on elements 0, 1, 2
        invrot = numpy.array(
            [[ 1.-2*s*(q1*q1+q2*q2) ,    2*s*(q0*q1+q2*q3) ,    2*s*(q0*q2-q1*q3)],
             [    2*s*(q0*q1-q2*q3) , 1.-2*s*(q0*q0+q2*q2) ,    2*s*(q1*q2+q0*q3)],
             [    2*s*(q0*q2+q1*q3) ,    2*s*(q1*q2-q0*q3) , 1.-2*s*(q0*q0+q1*q1)]],
            dtype=numpy.float32)
        sca = numpy.array( [[ self._scale[0], 0., 0., 0. ],
                            [ 0., self._scale[1], 0., 0. ],
                            [ 0., 0., self._scale[2], 0. ],
                            [ 0., 0., 0., 1.]], dtype=numpy.float32 )
        invsca = numpy.array( [[ 1./self._scale[0], 0., 0., 0. ],
                               [ 0., 1./self._scale[1], 0., 0. ],
                               [ 0., 0., 1./self._scale[2], 0. ],
                               [ 0., 0., 0., 1.]], dtype=numpy.float32 )
        # Turns out this is *slightly* faster tha numpy.identity(4, dtype=numpy.float32)
        rotation = numpy.array( [ [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] ], dtype=numpy.float32)
        rotation[0:3, 0:3] = rot.T
        invrotation = numpy.array( [ [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] ], dtype=numpy.float32)
        invrotation[0:3, 0:3] = invrot.T
        translation = numpy.array( [ [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] ], dtype=numpy.float32)
        translation[3, 0:3] = self._pos
        invtrans = numpy.array( [ [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] ], dtype=numpy.float32)
        invtrans[3, 0:3] = -self._pos
        mat = numpy.matmul(sca, rotation)
        mat = numpy.matmul(mat, translation)
        self.model_matrix[:] = mat
        mat = numpy.matmul(invtrans, invrotation)
        mat = numpy.matmul(mat, invsca)
        self.inverse_model_matrix[0:3, 0:3] = mat[0:3, 0:3].T
        # It was faster to construct the inverse manually here
        # self.inverse_model_matrix[0:3, 0:3] = numpy.linalg.inv(mat[0:3, 0:3]).T

        self.broadcast("update matrix")

    @property
    def visible(self):
        """Set to False to remove an object from the display, True to put it back."""
        return self._visible

    @visible.setter
    def visible(self, value):
        # sys.stderr.write("In visible setter\n")
        value = bool(value)
        if value == self._visible: return

        self._visible = value
        if value == True:
            self.context.add_object(self)
            if self._trail is not None:
                self._trail.visible = True
        else:
            # import pdb; pdb.set_trace()
            self.context.remove_object(self)
            if self._trail is not None:
                self._trail.visible = False

    @property
    def color(self):
        """The color of an object: (red, green, blue)"""
        return self._color[0:3]

    @color.setter
    def color(self, rgb):
        if len(rgb) != 3:
            sys.stderr.write("ERROR!  Need all of r, g, and b for color.\n")
            sys.exit(20)
        self._color[0:3] = numpy.array(rgb)
        self.broadcast("update color")

    @property
    def opacity(self):
        """Opacity is not implemented."""
        return self.color[3]

    @opacity.setter
    def opacity(self, alpha):
        self._color[3] = alpha
        self.update_colordata()
        self.broadcast("update color")

    @property
    def make_trail(self):
        """Set this to True to start leaving a thin trail behind the object as you move it."""
        return self._make_trail

    @make_trail.setter
    def make_trail(self, val):
        if not val:
            if self._make_trail:
                self.kill_trail()
                self._make_trail = False
        if val:
            if not self._make_trail:
                self.initialize_trail()
                self._make_trail = True

    @property
    def interval(self):
        """Only add a new trail segment after the object has been moved this many times."""
        return self._interval

    @interval.setter
    def interval(self, val):
        self._interval = val
        if self._nexttrail > self._interval:
            self._nexttrail = self._interval

    @property
    def retain(self):
        """Number of trail segments to keep."""
        return self._retain

    @retain.setter
    def retain(self, val):
        if val != self._retain:
            self._retain = val
            if self._make_trail:
                self.initialize_trail()
                
    def kill_trail(self):
        """(Internal, do not call.)"""
        if self._trail is not None:
            self._trail.visible = False
        self._trail = None

    def initialize_trail(self):
        """(Internal, do not call.)"""
        self.kill_trail()
        if self._trail_color is None:
            color = self.color
        else:
            color = self._trail_color
        self._trail = Curve( color=color, retain=self._retain, points=[ self._pos ],
                             radius=self._trail_radius)
        self._nexttrail = self._interval

    def clear_trail(self):
        self._trail.points = []
        
    def update_trail(self):
        """(Internal, do not call.)"""
        if not self._make_trail: return
        self._nexttrail -= 1
        if self._nexttrail <= 0:
            self._nexttrail = self._interval
            self._trail.add_point(self._pos)
            # self._trail.push_point(self._pos)

                
    def __del__(self):
        raise Exception("Rob, you really need to think about object deletion.")
        self.visible = False
        self.destroy()

    def destroy(self):
        pass

# =====================================================================
class Faces(GrObject):
    """A set of triangles.

    vertices must be a [3*faces, 3] numpy array.  Each of the elements
    along axis 0 is one vertex (with x, y, z indexed along axis 1).
    Three elements in a row specify a triangle.  Try to make it so that
    the outward face of the triangle is what you'd get from the
    right-hand-rule crossing the second minus first vertices with the
    third minus second.
    """

    def __init__(self, vertices, normals=None, smooth=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_triangles = vertices.shape[0]//3
        if ( len(vertices.shape) != 2 or
             vertices.shape[0] % 3 != 0 or
             vertices.shape[0] == 0 or
             vertices.shape[1] != 3 ):
            raise Exception("Faces requires a (3n, 3) numpy array.")

        self.vertexdata = numpy.ones(self.num_triangles * 3 * 4, dtype=numpy.float32)
        self.normaldata = numpy.zeros(self.num_triangles * 3 * 3, dtype=numpy.float32)

        if normals is not None:
            if normals.shape != vertices.shape:
                raise Exception("Faces must have (3n, 3) numpy array for both vertices and normals.")
            for i in range(3*self.num_triangles):
                self.vertexdata[4*i : 4*i+3] = vertices[i : i+3, :]
                self.normaldata[3*i : 3*i+3] = normaldata[i : i+3, :]
        else:
            if smooth:
                raise Exception("Faces: smooth isn't implemented.")
            else:
                for i in range(self.num_triangles):
                    self.vertexdata[3*4*i+0 : 3*4*i+3 ] = vertices[3*i  , :]
                    self.vertexdata[3*4*i+4 : 3*4*i+7 ] = vertices[3*i+1, :]
                    self.vertexdata[3*4*i+8 : 3*4*i+11] = vertices[3*i+2, :]
                    l1 = vertices[3*i+1, :] - vertices[3*i, :]
                    l2 = vertices[3*i+2, :] - vertices[3*i+1, :]
                    norm = numpy.array( [ l1[1]*l2[2] - l1[2]*l2[1],
                                          l1[2]*l2[0] - l1[0]*l2[2],
                                          l1[0]*l2[1] - l1[1]*l2[0] ],
                                        dtype=numpy.float32 )
                    normnorm = math.sqrt( norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2] )
                    if normnorm < 1e-6:
                        import pdb; pdb.set_trace()
                        raise Exception("Faces error: degenerate triangle.")
                    norm /= normnorm
                    self.normaldata[3*3*i   : 3*3*i+3] = norm
                    self.normaldata[3*3*i+3 : 3*3*i+6] = norm
                    self.normaldata[3*3*i+6 : 3*3*i+9] = norm

        self.finish_init()

    def destroy(self):
        raise Exception("OMG ROB!  You need to figure out how to destroy things!")
        

# ======================================================================

class Box(GrObject):

    """A rectangular solid with dimenions (x,y,z) = (length,height,width)"""
    
    @staticmethod
    def make_box_buffers(context):
        with Subject._threadlock:
            if not hasattr(Box, "_box_vertices"):
                Box._box_vertices = numpy.array( [ -0.5, -0.5,  0.5, 1.,
                                                   -0.5, -0.5, -0.5, 1.,
                                                    0.5, -0.5,  0.5, 1.,
                                                    0.5, -0.5,  0.5, 1.,
                                                   -0.5, -0.5, -0.5, 1.,
                                                    0.5, -0.5, -0.5, 1.,

                                                   -0.5,  0.5,  0.5, 1.,
                                                    0.5,  0.5,  0.5, 1.,
                                                   -0.5,  0.5, -0.5, 1.,
                                                   -0.5,  0.5, -0.5, 1.,
                                                    0.5,  0.5,  0.5, 1.,
                                                    0.5,  0.5, -0.5, 1.,

                                                   -0.5, -0.5, -0.5, 1.,
                                                   -0.5, -0.5,  0.5, 1.,
                                                   -0.5,  0.5, -0.5, 1.,
                                                   -0.5,  0.5, -0.5, 1.,
                                                   -0.5, -0.5,  0.5, 1.,
                                                   -0.5,  0.5,  0.5, 1.,

                                                    0.5,  0.5, -0.5, 1.,
                                                    0.5,  0.5,  0.5, 1.,
                                                    0.5, -0.5, -0.5, 1.,
                                                    0.5, -0.5, -0.5, 1.,
                                                    0.5,  0.5,  0.5, 1.,
                                                    0.5, -0.5,  0.5, 1.,

                                                   -0.5, -0.5,  0.5, 1.,
                                                    0.5, -0.5,  0.5, 1.,
                                                   -0.5,  0.5,  0.5, 1.,
                                                   -0.5,  0.5,  0.5, 1.,
                                                    0.5, -0.5,  0.5, 1.,
                                                    0.5,  0.5,  0.5, 1.,

                                                    0.5, -0.5, -0.5, 1.,
                                                   -0.5, -0.5, -0.5, 1.,
                                                    0.5,  0.5, -0.5, 1.,
                                                    0.5,  0.5, -0.5, 1.,
                                                   -0.5, -0.5, -0.5, 1.,
                                                   -0.5,  0.5, -0.5, 1. ],
                                                 dtype = numpy.float32 )

                Box._box_normals = numpy.array( [  0., -1., 0., 0., -1., 0., 0., -1., 0.,
                                                   0., -1., 0., 0., -1., 0., 0., -1., 0.,

                                                   0., 1., 0., 0., 1., 0., 0., 1., 0.,
                                                   0., 1., 0., 0., 1., 0., 0., 1., 0.,

                                                   -1., 0., 0., -1., 0., 0., -1., 0., 0.,
                                                   -1., 0., 0., -1., 0., 0., -1., 0., 0.,

                                                   1., 0., 0., 1., 0., 0., 1., 0., 0.,
                                                   1., 0., 0., 1., 0., 0., 1., 0., 0.,

                                                   0., 0., 1., 0., 0., 1., 0., 0., 1.,
                                                   0., 0., 1., 0., 0., 1., 0., 0., 1.,

                                                   0., 0., -1., 0., 0., -1., 0., 0., -1.,
                                                   0., 0., -1., 0., 0., -1., 0., 0., -1. ],
                                                dtype = numpy.float32 )


    def __init__(self, length=1., width=1., height=1., *args, **kwargs):
        """Parameters:

        length — The size of the box along the x-axis (left-right with default camera orientation)
        height — The size of the box along the y-axis (up-down with default camera orientation)
        width — The size of the box along the z-axis (in-out with default camera orientatino)

        Plus standard GrObject parameters: context, pos, axis, up, scale, color, make_trail, interval, retain
        """

        super().__init__(*args, **kwargs)

        Box.make_box_buffers(self.context)

        self.num_triangles = 12
        self.vertexdata = Box._box_vertices
        self.normaldata = Box._box_normals

        self.length = length
        self.width = width
        self.height = height

        self.finish_init()

    @property
    def length(self):
        return self.sx

    @length.setter
    def length(self, value):
        self.sx = value

    @property
    def width(self):
        return self.sz

    @width.setter
    def width(self, value):
        self.sz = value

    @property
    def height(self):
        return self.sy

    @height.setter
    def height(self, value):
        self.sy = value

    def destroy(self):
        raise Exception("OMG ROB!  You need to figure out how to destroy things!")

# ======================================================================

class Tetrahedron(Faces):
    """A tetrahedron"""

    def __init__(self, *args, **kwargs):
        norm = 2*math.sqrt(2/3)
        sqrt6 = math.sqrt(6)
        sqrt3 = math.sqrt(3)
        sqrt2 = math.sqrt(2)
        verts = numpy.array( [ [ norm*sqrt3/(2*sqrt2) , 0 , 0, ],
                               [ -norm/(2*sqrt6), norm/2, -norm/(2*sqrt3), ],
                               [ -norm/(2*sqrt6), 0, norm/sqrt3, ],

                               [ norm*sqrt3/(2*sqrt2) , 0 , 0, ],
                               [ -norm/(2*sqrt6), 0, norm/sqrt3, ],
                               [ -norm/(2*sqrt6), -norm/2., -norm/(2*sqrt3), ],
                                                                   
                               [ norm*sqrt3/(2*sqrt2) , 0 , 0, ],
                               [ -norm/(2*sqrt6), -norm/2., -norm/(2*sqrt3), ],
                               [ -norm/(2*sqrt6), norm/2, -norm/(2*sqrt3), ],
                               
                               [ -norm/(2*sqrt6), norm/2, -norm/(2*sqrt3), ],
                               [ -norm/(2*sqrt6), -norm/2, -norm/(2*sqrt3), ],
                               [ -norm/(2*sqrt6), 0, norm/sqrt3 ] ]
                             ,dtype=numpy.float32 )

        super().__init__(verts, smooth=False, *args, **kwargs)
        
# ======================================================================

class Octahedron(Faces):
    """An octahedron"""

    def __init__(self, *args, **kwargs):
        verts = numpy.array( [ [ 0.,  1.,  0.,], [ 0.,  0.,  1.], [ 1.,  0.,  0.], 
                               [ 0.,  1.,  0.,], [-1.,  0.,  0.], [ 0.,  0.,  1.], 
                               [ 0.,  1.,  0.,], [ 0.,  0., -1.], [-1.,  0.,  0.], 
                               [ 0.,  1.,  0.,], [ 1.,  0.,  0.], [ 0.,  0., -1.], 
                               [ 0., -1.,  0.,], [ 1.,  0.,  0.], [ 0.,  0.,  1.], 
                               [ 0., -1.,  0.,], [ 0.,  0.,  1.], [-1.,  0.,  0.], 
                               [ 0., -1.,  0.,], [-1.,  0.,  0.], [ 0.,  0., -1.], 
                               [ 0., -1.,  0.,], [ 0.,  0., -1.], [ 1.,  0.,  0.] ],
                             dtype=numpy.float32 )
        super().__init__(verts, smooth=False, *args, **kwargs)

# ======================================================================

class Dodecahedron(Faces):
    """A dodecahedron"""

    classinit = False
    facepoints = None
    
    @staticmethod
    def doclassinit():
        if Dodecahedron.classinit: return
        
        # This may seem overdone; why not just a table of point
        #  positions?  I do not have a good answer to that question.
        points = numpy.zeros([20, 3])
        c72 = math.cos(72./180.*math.pi)
        s72 = math.sin(72./180.*math.pi)
        rot72z = numpy.array( [ [ c72, s72, 0 ],
                                [-s72, c72, 0 ],
                                [   0,   0, 1 ] ] )
        # Top 5, each edge rotated 72⁰ relative to the previous
        points[0] = [0., 0., 0.]
        points[1] = [1., 0., 0.]
        points[2] = points[1] + numpy.matmul(rot72z, points[1]-points[0])
        points[3] = points[2] + numpy.matmul(rot72z, points[2]-points[1])
        points[4] = points[3] + numpy.matmul(rot72z, points[3]-points[2])

        # Point 5 is the next one down connected to point 0.  It satisfies (given p0 is at 0):
        #  (p5-p0)·(p5-p0) = 1         x² + y² + z² = 1
        #  (p5-p0)·(p0-p4) = cos(72⁰)  -x*p04x - y*p04y = cos(72⁰)
        #  (p5-p0)·(p0-p1) = cos(72⁰)  x = -cos(72⁰)
        points[5][0] = -c72
        points[5][1] = (c72 - c72*points[4][0]) / (-points[4][1])
        points[5][2] =  -math.sqrt(1. - points[5][0]**2 - points[5][1]**2)

        # Points 6 through 9 are just rotated by successive 72⁰ and hanging off of succesive points
        diff = numpy.matmul(rot72z, points[5])
        points[6] = points[1] + diff
        diff = numpy.matmul(rot72z, diff)
        points[7] = points[2] + diff
        diff = numpy.matmul(rot72z, diff)
        points[8] = points[3] + diff
        diff = numpy.matmul(rot72z, diff)
        points[9] = points[4] + diff

        # Recenter all of these points
        cm = points[0:10, :].sum(axis=0)/10.
        points[0:10, :] -= cm

        # The next 10 points relative to each other is just a 180⁰ rotation about x
        points[10:20, 0] = points[0:10, 0]
        points[10:20, 1:3] = -points[0:10, 1:3]

        # point 18 is connected to points 5 and 6. Satisfies:
        # (p18-p5)·(p5-p0) = cos(72⁰)
        # Current p18 of offset incorrectly in z
        # (p18x - p5x)*(p5x-p0x) + (p18y - p5y)*(p5y-p0y) + (p18z+zoff - p5z)*(p5z-p0z) = cos(72⁰)
        # zoff = ( cos(72⁰) - (p18x-p5x)*(p5x-p0x) - (p18y-p5y)*(p5y-p0x) ) / (p5z-p0z) + p5z - p18z
        zoff = ( c72 -
                 (points[18][0]-points[5][0]) * (points[5][0]-points[0][0]) -
                 (points[18][1]-points[5][1]) * (points[5][1]-points[0][1])
                 ) / (points[5][2]-points[0][2]) + points[5][2] - points[18][2]
        points[10:20, 2] += zoff

        # Recenter again
        cm = points.sum(axis=0)/20.
        points -= cm

        # Renormalize; make all points dist 1 from origin
        mag = math.sqrt( numpy.square(points[0, :]).sum() )
        points /= mag
    
        # (Note: pentagon normals go the wrong way)
        # Edges and Faces: 0-1, 1-2, 2-3, 3-4, 4-0
        #                  0-5, 5-18, 18-6, 6-1, 1-0
        #                  1-6, 6-17, 17-7, 7-2, 2-1
        #                  2-7, 7-16, 16-8, 8-3, 3-2
        #                  3-8, 8-15, 15-9, 9-4, 4-3
        #                  4-9, 9-19, 19-5, 5-0, 0-4
        #                  10-15, 15-8, 8-16, 16-11, 11-10
        #                  11-16, 16-7, 7-17, 17-12, 12-11
        #                  12-17, 17-6, 6-18, 18-13, 13-12
        #                  13-18, 18-5, 5-19, 19-14, 14-13
        #                  14-19, 19-9, 9-15, 15-10, 10-14
        #                  10-11, 11-12, 12-13, 13-14, 14-15
        #
        # Break ABCDE pentagons into triangles as:
        # CBA, CAD, DAE

        facepoints = numpy.array( [ [ 0, 1, 2, 3, 4 ],
                                    [ 0, 5, 18, 6, 1],
                                    [ 1, 6, 17, 7, 2],
                                    [ 2, 7, 16, 8, 3],
                                    [ 3, 8, 15, 9, 4],
                                    [ 4, 9, 19, 5, 0],
                                    [ 10, 15, 8, 16, 11],
                                    [ 11, 16, 7, 17, 12],
                                    [ 12, 17, 6, 18, 13],
                                    [ 13, 18, 5, 19, 14],
                                    [ 14, 19, 9, 15, 10],
                                    [ 10, 11, 12, 13, 14] ] )

        nfaces = 12
        faces = numpy.empty( (nfaces, 3, 3, 3) )

        for face in range(nfaces):
            faces[face, 0, 0, :] = points[facepoints[face, 2]]
            faces[face, 0, 1, :] = points[facepoints[face, 1]]
            faces[face, 0, 2, :] = points[facepoints[face, 0]]
            faces[face, 1, 0, :] = points[facepoints[face, 2]]
            faces[face, 1, 1, :] = points[facepoints[face, 0]]
            faces[face, 1, 2, :] = points[facepoints[face, 3]]
            faces[face, 2, 0, :] = points[facepoints[face, 3]]
            faces[face, 2, 1, :] = points[facepoints[face, 0]]
            faces[face, 2, 2, :] = points[facepoints[face, 4]]
        faces.shape = (nfaces*3*3, 3)
        Dodecahedron.facepoints = faces
        Dodecahedron.classinit = True
        
        
    def __init__(self, *args, **kwargs):
        if not Dodecahedron.classinit:
            Dodecahedron.doclassinit()
        super().__init__(Dodecahedron.facepoints, smooth=False, *args, **kwargs)


# ======================================================================
# The Icosahedron code may seem a little over-verbose.  I originally
# wrote it using indexed buffers, back when I did one GL draw call for
# each object.  (Slow.)  Now that I dump all triangles into one giant
# VAO, I don't use indexed buffers.

class Icosahedron(GrObject):
    """An icosahedron or subdivided icosahedron, with flat or smooth shading (for spheres)."""
    
    @staticmethod
    def make_icosahedron_vertices(subdivisions=0):
        """Internal, do not call."""
        with Subject._threadlock:
            if not hasattr(Icosahedron, "_vertices"):
                Icosahedron._vertices = [None, None, None, None, None]
                Icosahedron._normals = [None, None, None, None, None]
                Icosahedron._flatnormals = [None, None, None, None, None]
                Icosahedron._numvertices = [None, None, None, None, None]

            if Icosahedron._vertices[subdivisions] is None:

                # sys.stderr.write("Creating icosahedron vertex data for {} subdivisions\n".format(subdivisions))

                vertices = numpy.zeros( 4*12, dtype=numpy.float32 )
                edges = numpy.zeros( (30, 2), dtype=numpy.uint16 )
                faces = numpy.zeros( (20, 3), dtype=numpy.uint16 )

                # Vertices: 1 at top (+x), 5 next row, 5 next row, 1 at bottom

                r = 1.0
                vertices[0:4] = [r, 0., 0., 1.]
                angles = numpy.arange(0, 2*math.pi, 2*math.pi/5)
                for i in range(len(angles)):
                    vertices[4+4*i:8+4*i] = [ 0.447213595499958*r,
                                              0.8944271909999162*r*math.cos(angles[i]),
                                              0.8944271909999162*r*math.sin(angles[i]),
                                              1.]
                    vertices[24+4*i:28+4*i] = [-0.447213595499958*r,
                                                0.8944271909999162*r*math.cos(angles[i]+angles[1]/2.),
                                                0.8944271909999162*r*math.sin(angles[i]+angles[1]/2.),
                                                1.]
                vertices[44:48] = [-r, 0., 0., 1.]

                edges[0:5, :]   = [ [0, 1], [0, 2], [0, 3], [0, 4], [0, 5] ]
                edges[5:10, :]  = [ [1, 2], [2, 3], [3, 4], [4, 5], [5, 1] ]
                edges[10:20, :] = [ [1, 6], [2, 6], [2, 7], [3, 7], [3, 8],
                                    [4, 8], [4, 9], [5, 9], [5, 10], [1, 10] ]
                edges[20:25, :] = [ [6, 7], [7, 8], [8, 9], [9, 10], [10, 6] ]
                edges[25:30, :] = [ [6, 11], [7, 11], [8, 11], [9, 11], [10, 11] ]

                faces[0:5, :] = [ [0, 5, 1], [1, 6, 2], [2, 7, 3], [3, 8, 4], [4, 9, 0] ]
                faces[5:10, :] = [ [5, 10, 11], [6, 12, 13], [7, 14, 15], [8, 16, 17],
                                   [9, 18, 19] ]
                faces[10:15, :] = [ [20, 12, 11], [21, 14, 13], [22, 16, 15],
                                    [23, 18, 17], [24, 10, 19] ]
                faces[15:20, :] = [ [25, 26, 20], [26, 27, 21], [27, 28, 22],
                                    [28, 29, 23], [29, 25, 24] ]

                for i in range(int(subdivisions)):
                    vertices, edges, faces = Icosahedron.subdivide(vertices, edges, faces, r)

                normals = numpy.zeros( 3*len(vertices)//4, dtype=numpy.float32 )
                for i in range(len(vertices)//4):
                    normals[3*i:3*i+3] = ( vertices[4*i:4*i+3] /
                                                math.sqrt( (vertices[4*i:4*i+3]**2).sum() ))

                flatnormals = numpy.zeros( 3 * faces.shape[0], dtype=numpy.float32 )
                for i in range(faces.shape[0]):
                    v1 = edges[faces[i, 0], 0]
                    v2 = edges[faces[i, 0], 1]
                    if ( edges[faces[i, 1], 0] == edges[faces[i, 0], 0] or
                         edges[faces[i, 1], 0] == edges[faces[i, 0], 1] ):
                        v3 = edges[faces[i, 1], 1]
                    else:
                        v3 = edges[faces[i, 1], 0]
                    x = vertices[4*v1]   + vertices[4*v2]   + vertices[4*v3]
                    y = vertices[4*v1+1] + vertices[4*v2+1] + vertices[4*v3+1]
                    z = vertices[4*v1+2] + vertices[4*v2+2] + vertices[4*v3+2]
                    vlen = math.sqrt(x*x + y*y + z*z)
                    flatnormals[3*i] = x/vlen
                    flatnormals[3*i+1] = y/vlen
                    flatnormals[3*i+2] = z/vlen
                        
                    
                rendervertices = numpy.zeros( 4*3*faces.shape[0], dtype=numpy.float32 )
                rendernormals = numpy.zeros( 3*3*faces.shape[0], dtype=numpy.float32 )
                renderflatnormals = numpy.zeros( 3*3*faces.shape[0], dtype=numpy.float32 )
                
                v = numpy.zeros(6, dtype=numpy.uint16)
                for i in range(faces.shape[0]):
                    dex = 0
                    for j in range(3):
                        renderflatnormals[ 3 * ( (i*3) + j ) + 0] = flatnormals[3*i]
                        renderflatnormals[ 3 * ( (i*3) + j ) + 1] = flatnormals[3*i+1]
                        renderflatnormals[ 3 * ( (i*3) + j ) + 2] = flatnormals[3*i+2]
                        for k in range(2):
                            v[dex] = edges[faces[i, j], k]
                            dex += 1

                    if len(numpy.unique(v)) != 3:
                        sys.stderr.write("ERROR with face {}, {} vertices: {}\n"
                                         .format(i, len(numpy.unique(v)), numpy.unique(v)))
                        sys.exit(20)

                    if ( ( edges[faces[i, 0], 0] == edges[faces[i, 1], 0] ) or
                         ( edges[faces[i, 0], 0] == edges[faces[i, 1], 1] ) ):
                        for k in range(4):
                            rendervertices[4*(3*i+0)+k] = vertices[4*edges[faces[i, 0], 1] + k]
                            rendervertices[4*(3*i+1)+k] = vertices[4*edges[faces[i, 0], 0] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+0)+k] = normals[3*edges[faces[i, 0], 1] + k]
                            rendernormals[3*(3*i+1)+k] = normals[3*edges[faces[i, 0], 0] + k]
                    else:
                        for k in range(4):
                            rendervertices[4*(3*i+0)+k] = vertices[4*edges[faces[i, 0], 0] + k]
                            rendervertices[4*(3*i+1)+k] = vertices[4*edges[faces[i, 0], 1] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+0)+k] = normals[3*edges[faces[i, 0], 0] + k]
                            rendernormals[3*(3*i+1)+k] = normals[3*edges[faces[i, 0], 1] + k]
                    if ( ( edges[faces[i, 1], 0] == edges[faces[i, 0], 0] ) or
                         ( edges[faces[i, 1], 0] == edges[faces[i, 0], 1] ) ):
                        for k in range(4):
                            rendervertices[4*(3*i+2)+k] = vertices[4*edges[faces[i, 1], 1] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+2)+k] = normals[3*edges[faces[i, 1], 1] + k]
                    else:
                        for k in range(4):
                            rendervertices[4*(3*i+2)+k] = vertices[4*edges[faces[i, 1], 0] + k]
                        for k in range(3):
                            rendernormals[3*(3*i+2)+k] = normals[3*edges[faces[i, 1], 0] + k]

                # sys.stderr.write("{} triangles, {} vertices, {} vertices in array\n"
                #                  .format(faces.shape[0], len(vertices)//4, len(rendervertices)//4))

                Icosahedron._vertices[subdivisions] = rendervertices
                Icosahedron._normals[subdivisions] = rendernormals
                Icosahedron._flatnormals[subdivisions] = renderflatnormals


    @staticmethod
    def subdivide(vertices, edges, faces, r=1.0):
        """Internal, do not call."""
        newverts = numpy.zeros( len(vertices) + 4*edges.shape[0], dtype=numpy.float32 )
        newverts[0:len(vertices)] = vertices
        numoldverts = len(vertices) // 4

        for i in range(edges.shape[0]):
            vertex = 0.5 * ( vertices[ 4*edges[i, 0] : 4*edges[i, 0]+4 ] +
                             vertices[ 4*edges[i, 1] : 4*edges[i, 1]+4 ] )
            vertex[0:3] *= r / math.sqrt( (vertex[0:3]**2).sum() )
            newverts[len(vertices) + 4*i : len(vertices) + 4*i + 4] = vertex

        newedges = numpy.zeros( (2*edges.shape[0] + 3*faces.shape[0], 2 ) ,
                                dtype=numpy.uint16 )
        newfaces = numpy.zeros( (4*faces.shape[0], 3) , dtype=numpy.uint16 )

        for en in range(edges.shape[0]):
            newedges[2*en, 0] = edges[en, 0]
            newedges[2*en, 1] = numoldverts+en
            newedges[2*en+1, 0] = edges[en, 1]
            newedges[2*en+1, 1] = numoldverts+en
        for fn in range(faces.shape[0]):
            newedges[2*edges.shape[0] + 3*fn + 0, 0] = numoldverts + faces[fn, 0]
            newedges[2*edges.shape[0] + 3*fn + 0, 1] = numoldverts + faces[fn, 1]
            newedges[2*edges.shape[0] + 3*fn + 1, 0] = numoldverts + faces[fn, 1]
            newedges[2*edges.shape[0] + 3*fn + 1, 1] = numoldverts + faces[fn, 2]
            newedges[2*edges.shape[0] + 3*fn + 2, 0] = numoldverts + faces[fn, 2]
            newedges[2*edges.shape[0] + 3*fn + 2, 1] = numoldverts + faces[fn, 0]

        for fn in range(faces.shape[0]):
            if ( edges[faces[fn, 0], 0] == edges[faces[fn, 1], 0] or
                 edges[faces[fn, 0], 0] == edges[faces[fn, 1], 1] ):
                corner1 = edges[faces[fn, 0], 1]
                corner2 = edges[faces[fn, 0], 0]
            else:
                corner1 = edges[faces[fn, 0], 0]
                corner2 = edges[faces[fn, 0], 1]
            if ( edges[faces[fn, 1], 0] == edges[faces[fn, 0], 0] or
                 edges[faces[fn, 1], 0] == edges[faces[fn, 0], 1] ):
                corner3 = edges[faces[fn, 1], 1]
            else:
                corner3 = edges[faces[fn, 1], 0]

            if newedges[2*faces[fn, 0], 0] == corner1:
                edge1l = 2*faces[fn, 0]
                edge1r = 2*faces[fn, 0] + 1
            else:
                edge1l = 2*faces[fn, 0] + 1
                edge1r = 2*faces[fn, 0]
            if newedges[2*faces[fn, 1], 0] == corner2:
                edge2l = 2*faces[fn, 1]
                edge2r = 2*faces[fn, 1] + 1
            else:
                edge2l = 2*faces[fn, 1] + 1
                edge2r = 2*faces[fn, 1]
            if newedges[2*faces[fn, 2], 0] == corner3:
                edge3l = 2*faces[fn, 2]
                edge3r = 2*faces[fn, 2] + 1
            else:
                edge3l = 2*faces[fn, 2] + 1
                edge3r = 2*faces[fn, 2]
            mid1 = 2*edges.shape[0] + 3*fn
            mid2 = 2*edges.shape[0] + 3*fn + 1
            mid3 = 2*edges.shape[0] + 3*fn + 2

            newfaces[4*fn,     :] = [edge1l, mid3, edge3r]
            newfaces[4*fn + 1, :] = [edge1r, edge2l, mid1]
            newfaces[4*fn + 2, :] = [mid2, edge2r, edge3l]
            newfaces[4*fn + 3, :] = [mid1, mid2, mid3]

        return (newverts, newedges, newfaces)


    def __init__(self, radius=1., flat=True, subdivisions=0, *args, **kwargs):
        """Parameters:

        radius — The radius of the icosahedron (to the points) or sphere (default: 1)
        flat — If True, render faces flat; otherwise, smooth shade them to approximate a sphere. (default: False)
        subdivisions — How many times to subdivide the icosahedron; each subdivision increases number of faces 4×
                       (default 0)

        Plus the usual GrObject parameters.
        """

        super().__init__(*args, **kwargs)

        if subdivisions > 4.:
            raise Exception(">4 subdivisions is absurd. Even 4 is probably too many!!!")

        Icosahedron.make_icosahedron_vertices(subdivisions)

        self.num_triangles = len(Icosahedron._vertices[subdivisions]) // 12
        self.vertexdata = Icosahedron._vertices[subdivisions]
        if flat:
            self.normaldata = Icosahedron._flatnormals[subdivisions]
        else:
            self.normaldata = Icosahedron._normals[subdivisions]

        self.radius = radius

        self.finish_init()

        # sys.stderr.write("Created icosahedron with {} triangles.\n".format(self.num_triangles))

    @property
    def radius(self):
        """The radius of the icosahedron or sphere."""
        return self.scale.sum()/3.

    @radius.setter
    def radius(self, r):
        self.scale = numpy.array( [r, r, r] )


class Sphere(Icosahedron):
    """A sphere, modelled by default as a 2x subdivided icosahedron (320 faces)."""
    def __init__(self, subdivisions=2, *args, **kwargs):
        """Parameters:

        subdivisions — higher = more faces (default: 2) (More than 3 is excessive, even 3 is probably excessive.)
        radius — radius of the sphere (default 1)

        Plus the usual GrObject parameters
        """
        super().__init__(subdivisions=subdivisions, flat=False, *args, **kwargs)


class Ellipsoid(Icosahedron):
    """An ellipsoid."""

    def __init__(self, subdivisions=2, length=1., width=1., height=1., *args, **kwargs):
        """Parameters:

        subdivisions — higher = more faces (default: 2, which is probably always good)
        length — diameter along x-axis (for unrotated object)
        width — diameter along z-axis (for unrotated object)
        height — diameter along y-axis (for unrotated object)

        Plus the usual GrObject parameters.
        """
        super().__init__(subdivisions=subdivisions, flat=False, *args, **kwargs)

        self.length = length
        self.width = width
        self.height = height
        
    @property
    def length(self):
        """The diameter along the object's local x-axis."""
        return self.sx/2.

    @length.setter
    def length(self, value):
        self.sx = value/2.

    @property
    def width(self):
        """The diameter along the object's local z-axis."""
        return self.sz/2.

    @width.setter
    def width(self, value):
        self.sz = value/2.

    @property
    def height(self):
        """The diameter along the object's local y-axis."""
        return self.sy/2.

    @height.setter
    def height(self, value):
        self.sy = value/2.

# # ======================================================================

class Cylinder(GrObject):
    """A cylinder, oriented by default along the x-axis."""
    
    @staticmethod
    def make_cylinder_vertices(num_edge_points=16):
        """Internal, do not call."""
        with Subject._threadlock:
            if not hasattr(Cylinder, "_vertices"):
                Cylinder._vertices = {}
                Cylinder._normals = {}

        with Subject._threadlock:
            if not num_edge_points in Cylinder._vertices:
                
                # Number of triangles = 4 * num_edge_points
                #   one set of num_edge_points for each cap
                #   two sets of num_edge_points for the two triangles on the sides

                vertices = numpy.empty( 4 * 3*4*num_edge_points, dtype=numpy.float32)
                normals = numpy.empty( 3 * 3*4*num_edge_points, dtype=numpy.float32)

                # Endcaps are at ±1

                # Make an array of phis to make sure identical floats show up
                #   where they're supposed to
                phis = numpy.arange(num_edge_points+1) * 2*math.pi / num_edge_points
                phis[num_edge_points] = 0.
                dphi = phis[1] / 2.
                
                # Endcaps
                off = 3*num_edge_points
                for i in range(num_edge_points):
                    vertices[4 * (3*i+0) : 4 * (3*i+0) + 4] = [1., 0., 0., 1.]
                    vertices[4 * (3*i+1) : 4 * (3*i+1) + 4] = [1., math.cos(phis[i]), math.sin(phis[i]), 1.]
                    vertices[4 * (3*i+2) : 4 * (3*i+2) + 4] = [1., math.cos(phis[i+1]), math.sin(phis[i+1]), 1.]
                    normals[3 * (3*i+0) : 3 * (3*i+0) + 3] = [1., 0., 0.]
                    normals[3 * (3*i+1) : 3 * (3*i+1) + 3] = [1., 0., 0.]
                    normals[3 * (3*i+2) : 3 * (3*i+2) + 3] = [1., 0., 0.]
                    vertices[4 * (off+(3*i+0)) : 4 * (off+(3*i+0)) + 4] = [0., 0., 0., 1]
                    vertices[4 * (off+(3*i+1)) : 4 * (off+(3*i+1)) + 4] = [0., math.cos(phis[i]), math.sin(phis[i]), 1.]
                    vertices[4 * (off+(3*i+2)) : 4 * (off+(3*i+2)) + 4] = [0., math.cos(phis[i+1]),
                                                                           math.sin(phis[i+1]), 1.]
                    normals[3 * (off+3*i+0) : 3 * (off+3*i+0) + 3] = [-1., 0., 0.]
                    normals[3 * (off+3*i+1) : 3 * (off+3*i+1) + 3] = [-1., 0., 0.]
                    normals[3 * (off+3*i+2) : 3 * (off+3*i+2) + 3] = [-1., 0., 0.]

                # Sides
                off = 6*num_edge_points
                for i in range(num_edge_points):
                    vertices[4 * (off + 6*i + 0) : 4 * (off + 6*i + 0) + 4] = [1., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    normals[3 * (off + 6*i + 0) : 3 * (off + 6*i + 0) + 3] = [0., math.cos(phis[i]),
                                                                              math.sin(phis[i])]
                    vertices[4 * (off + 6*i + 1) : 4 * (off + 6*i + 1) + 4] = [1., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    normals[3 * (off + 6*i + 1) : 3 * (off + 6*i + 1) + 3] = [0., math.cos(phis[i+1]),
                                                                              math.sin(phis[i+1])]
                    vertices[4 * (off + 6*i + 2) : 4 * (off + 6*i + 2) + 4] = [0., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    normals[3 * (off + 6*i + 2) : 3 * (off + 6*i + 2) + 3] = [0., math.cos(phis[i+1]),
                                                                              math.sin(phis[i+1])]

                    vertices[4 * (off + 6*i + 3) : 4 * (off + 6*i + 3) + 4] = [0., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    normals[3 * (off + 6*i + 3) : 3 * (off + 6*i + 3) + 3] = [0., math.cos(phis[i+1]),
                                                                              math.sin(phis[i+1])]
                    vertices[4 * (off + 6*i + 4) : 4 * (off + 6*i + 4) + 4] = [0., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    normals[3 * (off + 6*i + 4) : 3 * (off + 6*i + 4) + 3] = [0., math.cos(phis[i]),
                                                                              math.sin(phis[i])]
                    vertices[4 * (off + 6*i + 5) : 4 * (off + 6*i + 5) + 4] = [1., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    normals[3 * (off + 6*i + 5) : 3 * (off + 6*i + 5) + 3] = [0., math.cos(phis[i]),
                                                                              math.sin(phis[i])]
                    
                Cylinder._vertices[num_edge_points] = vertices
                Cylinder._normals[num_edge_points] = normals


    def __init__(self, radius=1., num_edge_points=16, *args, **kwargs):
        """Parameters:

        radius — The transverse radius of the cylinder.
        num_edge_points — How many faces the end polygons have (default: 16)
        (Use axis to set both the orientation and length of the cylinder.)

        Plus the usual GrObject parameters.
        """

        super().__init__(*args, **kwargs)

        Cylinder.make_cylinder_vertices(num_edge_points)

        self.vertexdata = Cylinder._vertices[num_edge_points]
        self.normaldata = Cylinder._normals[num_edge_points]
        self.num_triangles = len(Cylinder._vertices[num_edge_points]) // 12

        self.radius = radius

        # sys.stderr.write("Made cylinder with radius {} and {} triangles.\n".format(radius, self.num_triangles))
        
        self.finish_init()

    @property
    def radius(self):
        """Transverse radius of the cylinder."""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.scale = [self._scale[0], value, value]

# ======================================================================

class Cone(GrObject):
    """A cone with its base at the origin and its tip at 1,0,0"""
    
    @staticmethod
    def make_cone_vertices():
        """Internal, do not call."""
        with Subject._threadlock:
            if not hasattr(Cone, "_vertices"):
                num_edge_points = 16

                # Number of trianges = 2 * num_edge_points
                # one set for the base
                # one set for the edges
                #
                # Base is at origin, tip is at 1,0,0

                vertices = numpy.empty( 4 * 3*2*num_edge_points, dtype=numpy.float32)
                normals = numpy.empty( 3 * 3*4*num_edge_points, dtype=numpy.float32)

                # Make an array of phis to make sure identical floats show up
                #   where they're supposed to
                phis = numpy.arange(num_edge_points+1) * 2*math.pi / num_edge_points
                phis[num_edge_points] = 0.
                dphi = math.pi / num_edge_points

                # Endcap
                for i in range(num_edge_points):
                    vertices[4 * (3*i+0) : 4 * (3*i+0) + 4] = [0., 0., 0., 1.]
                    vertices[4 * (3*i+1) : 4 * (3*i+1) + 4] = [0., math.cos(phis[i]), math.sin(phis[i]), 1.]
                    vertices[4 * (3*i+2) : 4 * (3*i+2) + 4] = [0., math.cos(phis[i+1]), math.sin(phis[i+1]), 1.]
                    normals[3 * (3*i+0) : 3 * (3*i+0) + 3] = [-1., 0., 0.]
                    normals[3 * (3*i+1) : 3 * (3*i+1) + 3] = [-1., 0., 0.]
                    normals[3 * (3*i+2) : 3 * (3*i+2) + 3] = [-1., 0., 0.]

                # Edges
                off = 3*num_edge_points
                sqrt2 = math.sqrt(2.)
                for i in range(num_edge_points):
                    vertices[4 * (off + 3*i + 0) : 4 * (off + 3*i + 0) + 4] = [0., math.cos(phis[i]),
                                                                               math.sin(phis[i]), 1.]
                    vertices[4 * (off + 3*i + 1) : 4 * (off + 3*i + 1) + 4] = [0., math.cos(phis[i+1]),
                                                                               math.sin(phis[i+1]), 1.]
                    vertices[4 * (off + 3*i + 2) : 4 * (off + 3*i + 2) + 4] = [1., 0., 0., 1.]

                    
                    normals[3 * (off + 3*i + 0) : 3 * (off + 3*i + 0) + 3] = [1./sqrt2, math.cos(phis[i])/sqrt2,
                                                                              math.sin(phis[i])/sqrt2]
                    normals[3 * (off + 3*i + 1) : 3 * (off + 3*i + 1) + 3] = [1./sqrt2, math.cos(phis[i+1])/sqrt2,
                                                                              math.sin(phis[i+1])/sqrt2]
                    normals[3 * (off + 3*i + 2) : 3 * (off + 3*i + 2) + 3] = [1., 0., 0.]

                Cone._vertices = vertices
                Cone._normals = normals


    def __init__(self, radius=1., *args, **kwargs):
        """Parameters:

        radius — the radius of the circle that makes the base of the cone.  (Really a 12-sided polygon, smooth shaded.)

        Use the axis parmaeter to adjust the orientation and length of the cone.

        Plus the usual GrObject parameters.
        """
        
        super().__init__(*args, **kwargs)

        Cone.make_cone_vertices()

        self.vertexdata = Cone._vertices
        self.normaldata = Cone._normals
        self.num_triangles = len(self.vertexdata) // 12

        self.radius = radius

        # sys.stderr.write("Made cone with radius {} and {} triangles.\n".format(radius, self.num_triangles))
        
        self.finish_init()

    @property
    def radius(self):
        """The radius of the base of the cone."""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self.scale = [self._scale[0], value, value]

# ======================================================================

class Arrow(GrObject):
    """An arrow with a square cross-section shaft."""

    def __init__(self, shaftwidth=None, headwidth=None, headlength=None, fixedwidth=False, *args, **kwargs):
        """Parmaeters:
        
        shaftwidth — The width of the shaft (default: 0.1 * length)
        headwidth — The width of the head (default: 2 * shaftwidth)
        headlength — The length of the head (default: 3 * shaftidth)
        fixedwidth — If False (default), all of the dimensions scale when you change the length of the arrow.
                     If True then shaftwidth, headwidth, and headlength stays fixed.
        
        Control the orientation and length of the arrow with axis.

        Plus the usual GrObject parameters.
        """

        super().__init__(*args, **kwargs)

        length = self.sx
        if shaftwidth is None:
            shaftwidth = 0.1 * length
        if headwidth is None:
            headwidth = 2 * shaftwidth
        if headlength is None:
            headlength = 3 * shaftwidth

        if not fixedwidth:
            headlength /= shaftwidth
            headwidth /= shaftwidth
            shaftwidth /= length
        else:
            # Because we're using scaling, I have to divide out the length
            # ROB THINK ABOUT ALL THIS
            shaftwidth /= length
            headwidth /= length
            
        self.fixedwidth = fixedwidth
        self.shaftwidth = shaftwidth
        self.headwidth = headwidth
        self.headlength = headlength

        self.make_arrow()

        self.finish_init()

    def make_arrow(self):
        """Internal, do not use."""
        # 16 triangles:
        #   2 for the base
        #   2 for each side (8 total)
        #   2 for the bottom of the head
        #   4 for the head
        self.num_triangles = 16
        self.vertexdata = numpy.empty(4 * 16 * 3, dtype=numpy.float32)
        self.normaldata = numpy.empty(3 * 16 * 3, dtype=numpy.float32)

        length = self.sx
        # Base
        if self.fixedwidth:
            shaftw = self.shaftwidth
            headw = self.headwidth
            headl = self.headlength
        else:
            shaftw = self.shaftwidth * length
            headw = self.headwidth * shaftw
            headl = self.headlength * shaftw

        # The length will get scaled by sx, so I have to renormalize
        # here.  (This is kind of messy, with sx having a special
        # meaning.  Side effect of inheritance, I guess.)
            
        headl /= length
        if (headl > 0.5): headl = 0.5
        shaftl = 1. - headl
            
        sys.stderr.write("length={:.3f}, shaftl={:.3f}, shaftw={:.3f}, headw={:.3f}, headl={:.3f}\n"
                         .format(length, shaftl, shaftw, headw, headl))

        # Base
        self.vertexdata[0:3*2*4] = [0., -shaftw/2., shaftw/2., 1.,
                                    0., shaftw/2., -shaftw/2., 1.,
                                    0., shaftw/2., shaftw/2., 1.,
                                    
                                    0., -shaftw/2., shaftw/2., 1.,
                                    0., -shaftw/2., -shaftw/2., 1.,
                                    0., shaftw/2., -shaftw/2., 1.]
        self.normaldata[0:6*3] = [-1., 0., 0., -1., 0., 0., -1., 0., 0.,
                                  -1., 0., 0., -1., 0., 0., -1., 0., 0.]

        # Sides
        firstcorner = numpy.array([ [1, -1],  [1, 1],  [-1, 1], [-1, -1] ], dtype=numpy.int32)
        secondcorner = numpy.array([ [-1, -1], [1, -1], [1, 1],  [-1, 1] ], dtype=numpy.int32)
        normalvals = numpy.array([ [0., -1.], [1., 0.], [0., 1.], [-1., 0.] ],dtype=numpy.float32)
        off = 2
        for j in range(4):
            self.vertexdata[4 * (3 * (off + 2*j)) + 0 : 4 * (3 * (off + 2*j)) + 4] = [0.,
                                                                                      firstcorner[j, 0]*shaftw/2,
                                                                                      firstcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) + 4 : 4 * (3 * (off + 2*j)) + 8] = [0.,
                                                                                      secondcorner[j, 0]*shaftw/2,
                                                                                      secondcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) + 8 : 4 * (3 * (off + 2*j)) +12] = [shaftl,
                                                                                      secondcorner[j, 0]*shaftw/2,
                                                                                      secondcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) +12 : 4 * (3 * (off + 2*j)) +16] = [shaftl,
                                                                                      secondcorner[j, 0]*shaftw/2,
                                                                                      secondcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) +16 : 4 * (3 * (off + 2*j)) +20] = [shaftl,
                                                                                      firstcorner[j, 0]*shaftw/2,
                                                                                      firstcorner[j, 1]*shaftw/2,
                                                                                      1.]
            self.vertexdata[4 * (3 * (off + 2*j)) +20 : 4 * (3 * (off + 2*j)) +24] = [0.,
                                                                                      firstcorner[j, 0]*shaftw/2,
                                                                                      firstcorner[j, 1]*shaftw/2,
                                                                                      1.]
            for k in range(6):
                self.normaldata[3 * (3 * (off + 2*j) + k) + 0 :
                                3 * (3 * (off + 2*j) + k) + 3] = [0., normalvals[j,0], normalvals[j, 1]]
            
        # Base of head
        off = 10
        self.vertexdata[ 4 * (3*(off + 0)) : 4 * (3*(off + 2))] = [shaftl, -headw/2., headw/2., 1.,
                                                                   shaftl, headw/2., -headw/2., 1.,
                                                                   shaftl, headw/2., headw/2., 1.,
                                                                   
                                                                   shaftl, -headw/2., headw/2., 1.,
                                                                   shaftl, -headw/2., -headw/2., 1.,
                                                                   shaftl, headw/2., -headw/2., 1.]
        self.normaldata[ 3 * (3*(off + 0)) : 3 * (3*(off + 2))] = [-1., 0., 0., -1., 0., 0., -1., 0., 0.,
                                                                   -1., 0., 0., -1., 0., 0., -1., 0., 0.]
        

        # Head
        off = 12
        yzlen = headl / math.sqrt(headl*headl + (headw/2.)*(headw/2.))
        xlen = math.sqrt(1. - yzlen*yzlen)
        for j in range(4):
            self.vertexdata[ 4 * (3*(off+j)) :
                             4 * (3*(off+j + 1)) ] = [shaftl, firstcorner[j, 0]*headw/2, firstcorner[j, 1]*headw/2, 1.,
                                                      shaftl, secondcorner[j, 0]*headw/2, secondcorner[j, 1]*headw/2, 1.,
                                                      1., 0., 0., 1.]
            self.normaldata[ 3 * (3*(off+j)) :
                             3 * (3*(off +j + 1)) ] = [xlen, normalvals[j, 0]*yzlen, normalvals[j, 1]*yzlen,
                                                       xlen, normalvals[j, 0]*yzlen, normalvals[j, 1]*yzlen,
                                                       1., 0., 0]

    @GrObject.axis.setter
    def axis(self, value):
        """Orient and scale the overall length of arrow.

        Its other dimenions change based on whether or not you specified fixedwidth=True when creating the arrow.
        """
        GrObject.axis.fset(self, value)
        if hasattr(self, 'fixedwidth') and self.fixedwidth:
            self.make_arrow()
            self.broadcast("update vertices")
        else:
            # I'm not completely happy about this, because the call to
            # fset(self, axis) and also setting scale means that
            # update_model_matrix gets called twice; we really should
            # have been able to skip the first one.  Add a parameter to
            # the GrObject.axis setter?  Seems janky
            length = math.sqrt(value[0]*value[0] + value[1]*value[1] + value[2]*value[2])
            self.scale = [length, length, length]
                
# ======================================================================
# This is a curve intended when you're going to update the points a lot.
# The full triangles are actually caulcated in a geometry shader, which
# will be a lot faster than calculating them in python...  but they get
# redone every bloody frame.  This is reaosnable if the curve's points
# are being updated a lot, as the calculations will have to be redone a
# lot as is.  If the curve's points are hardly ever updated, then it's
# much better to use another class that I haven't written yet....

class Curve(GrObject):
    """A curved tube around a path.  The path is specified by a series of points.
       It keeps at most "retain" points, pulling things off the front if you add
       more to the end.
    """

    _TOO_CLOSE = 1e-8
    
    def __init__(self, radius=0.05, points=None, retain=150, *args, **kwargs):
        """Parameters:

        points — A n×3 array that specifies the position of n points.
                 This is the path that is the center of the tube.
        radius — The radius of the tube that will be drawn around the path.
        retain — Max number of points to retain (default: 150)

        Plus the other standard GrObject parameters.
        """
        
        super().__init__(*args, **kwargs)

        self._object_type = GLObjectCollection._OBJ_TYPE_CURVE

        if retain < 3:
            raise Exception("C'mon, retain at least THREE points.  Geez.")
        self._retain = retain
        self._radius = radius
        
        self._points = numpy.zeros( (retain, 3), dtype=numpy.float32 )
        self._transverse = numpy.zeros( (retain, 3), dtype=numpy.float32 )
        self._numpoints = 0

        if points is not None:
            points = numpy.array(points, dtype=numpy.float32)
            if len(points.shape) == 1 and points.shape[0] == 3:
                points = numpy.array( [points], dtype=numpy.float32)
            if len(points.shape) != 2 or points.shape[1] != 3:
                raise Exception("To make a curve, points must be n×3.")

        if points is not None and points.shape[0] > 0:
            # Import the points, skipping repeated points
            tooclose = Curve._TOO_CLOSE**2
            self._points[0, :] = points[0, :]
            self._numpoints = 1
            for i in range(1, points.shape[0]):
                mag = numpy.square(points[i,:] - points[i-1, :]).sum()
                if mag > tooclose:
                    self._points[self._numpoints, :] = points[i, :]
                    self._numpoints += 1
                    
            #  self._points = numpy.array(newpoints, dtype=numpy.float32)
            if self._numpoints > retain:
                raise Exception("Dude, don't make retain less than the number of initial points.")
        
            self.make_transverse()

        self.finish_init()

    # Adds a point to the end of the curve
    # Yoink one from the front if the total number is beyond retain
    def add_point(self, pos):
        """Add a new point to the end of the curve, and remove the first point from the curve.

        pos — The position of the new point.
        """

        lengthchanged = False
        pos = numpy.array(pos)
        if pos.shape != (3,):
            raise Exception("Must pass 3 elements to add_point, you passed {}".format(pos.shape))

        if self._numpoints == 0:
            self._points[0, :] = pos
            self._numpoints += 1
            self._transverse[0, :] = [0., 0., self.radius ]
            if self._visible: self.broadcast("yank and readd")
            return

        axishat = numpy.array(pos) - numpy.array(self._points[-1])
        magaxis = math.sqrt( numpy.square(axishat).sum() )
        if magaxis < Curve._TOO_CLOSE:
            return

        if self._numpoints == 1:
            self._points[1, :] = pos
            self._numpoints += 1
            self.make_transverse()
            if self._visible: self.broadcast("yank and readd")

        else:
            if self._numpoints >= self.retain:
                # is this safe?
                self._points[:-1, :] = self._points[1:, :]
                self._transverse[:-1, :] = self._transverse[1:, :]
            else:
                self._numpoints += 1
                lengthchanged = True

            self._points[self._numpoints-1, :] = pos
            self.make_transverse(startat=self._numpoints-2)

            if lengthchanged:
                if self._visible: self.broadcast("yank and readd")
            else:
                self.broadcast("update vertices")

    @property
    def numpoints(self):
        """Number of points currently on the curve."""
        return self._numpoints

    @property
    def retain(self):
        """Number of points that are kept.  Can't be changed after initialization."""
        return self._retain
    
    @property
    def points(self):
        """The array of points on the curve.  Returned by reference, I think, so be careful."""
        return self._points[0:self._numpoints, :]

    @points.setter
    def points(self, points):
        if points is None or len(points) == 0:
            self._numpoints = 0
            if self._visible: self.broadcast("yank and readd")
            return

        if len(points.shape) != 2 or points.shape[1] != 3:
            raise Exception("Illegal points; must be n×3.")
        if points.shape[0] > self.retain:
            raise Exception("Tried to set points to an array longer than retain.")

        self._points[0:points.shape[0], :] = points
        numchanged = False
        if points.shape[0] != self._numpoints:
            self._numpoints = points.shape[0]
            numchanged = True
        self.make_transverse()
        if numchanged:
            if self._visible: self.broadcast("yank and readd")
        else:
            self.broadcast("update vertices")

    @property
    def trans(self):
        """The array of transverse vectors that were generated for the curve."""
        return self._transverse[0:self._numpoints, :]
        
    @property
    def radius(self):
        """The radius of the tube around the path."""
        return self._radius

    @radius.setter
    def radius(self, rad):
        if self._radius != rad:
            self._transverse *= rad / self._radius
            self._radius = rad
            self.broadcast("update vertices")

    def make_transverse(self, startat=0):
        """Internal, do not call."""

        num = self._numpoints
        axes = self._points[1:num, :] - self._points[:num-1, :]
        axesmag = numpy.sqrt(numpy.square(axes).sum(axis=1))
        # Note: this will div by 0 if any points are doubled
        hatxes = axes / axesmag[:, numpy.newaxis]

        toosmalltransmag = 1e-6

        if num == 0:
            return
        
        if num == 1:
            self._transverse[0, :] = [0., 0., self.radius]
            return
        
        if num > 2:
            # All points but first and last, the transverse
            #   is just along the difference of the unit vectors
            #   along the two directions of the previous and next.
            #   This is perpendicular to the "tangent" of the
            #   curve approximated by that corner.

            self._transverse[startat+1:num-1, :] = hatxes[startat+1:, :] - hatxes[startat:-1, :]

            # First and last points : take the adjacent transverse, but then only
            # the component perpendicular to the one axis it's sticking to

            if startat > 0:
                self._transverse[startat, :] = hatxes[startat, :] - hatxes[startat-1, :]
            else:
                self._transverse[0, : ] = self._transverse[1, :] - hatxes[0] * (self._transverse[1, :] *
                                                                                hatxes[0]).sum()
            self._transverse[num-1, :] = self._transverse[num-2, :] - hatxes[-1] * (self._transverse[num-2, :] *
                                                                                  hatxes[-1]).sum()

            transmag = numpy.sqrt(numpy.square(self._transverse[:num]).sum(axis=1))

            # Special case problem... if transmag[0] is 0, then we have to search forward
            #  for the first transmag that's not 0, and copy all the rest back.  Then, only
            #  keep the part that's perpendicular to the axis
            if (startat == 0) and (transmag[0] < toosmalltransmag):
                w = numpy.where(transmag >= toosmalltransmag)
                if len(w[0]) == 0:
                    # This is a cylinder, not a curve... go back to the 2-point solution
                    # if axis isn't along z, cross z with it get transverse.  Otherwise, cross x with it
                    if hatxes[0, 2] < 0.9:
                        self._transverse[:num, :] = numpy.array( [ -hatxes[0, 1], hatxes[0, 0], 0. ],
                                                                 dtype=numpy.float32 )
                    else:
                        self._transverse[:num, :] = numpy.array( [ 0., -hatxes[0, 2], hatxes[0, 1] ],
                                                                 dtype=numpy.flat32 )
                else:
                    newtrans = self._transverse[w[0][0], :]
                    newtrans -= hatxes[0] * (newtrans * hatxes[0]).sum()
                    self._transverse[:w[0][0]-1, :] = self._transverse[w[0][0], :]


            transmag = numpy.sqrt(numpy.square(self._transverse[startat:num]).sum(axis=1))

            # Now find all places where transverse is still 0, and make those
            #  the same as previous
            w = numpy.where( transmag < toosmalltransmag )
            # (Doing a for loop so that if there are multiple 0s in a row, the
            #   copy propagates forward)
            for i in w[0]:
                self._transverse[startat+i, :] = self._transverse[startat+i-1, :]
                self._transverse[startat+i, :] -= ( (hatxes[startat+i-1]*self._transverse[startat+i]).sum() *
                                                    hatxes[startat+i-1] )
                transmag[i] = math.sqrt( numpy.square(self._transverse[startat+i, :]).sum() )

            # Figure out where we have a >90⁰ rotation between two transverses,
            #   and flip all transverses after that to fix this.
            if startat == 0:
                transversedot = ( self._transverse[:num-1, :] * self._transverse[1:num, :] ).sum(axis=1)
                wflip = numpy.where(transversedot < 0.)
                for i in wflip[0]:
                    self._transverse[i+1:num, :] *= -1.
            else:
                transversedot = ( self._transverse[startat-1:num-1, :] * self._transverse[startat:num, :]).sum(axis=1)
                wflip = numpy.where(transversedot < 0.)
                for i in wflip[0]:
                    self._transverse[startat+i:num, :] *= -1.

            self._transverse[startat:num, :] *= self._radius / transmag[:, numpy.newaxis]
        else:
            # Just 2 points
            # if axis isn't along z, cross z with it get transverse.  Otherwise, cross x with it
            if hatxes[0, 2] < 0.9:
                self._transverse[0:1, :] = numpy.array( [ -hatxes[0, 1], hatxes[0, 0], 0. ], dtype=numpy.float32 )
            else:
                self._transverse[0:1, :] = numpy.array( [ 0., -hatxes[0, 2], hatxes[0, 1] ], dtype=numpy.float32 )
            transmag = numpy.sqrt( numpy.square(self._transverse[0:1, :].sum(axis=1)) )
            self._transverse[0:1, :] *= self.radius / transmag[:, numpy.newaxis]

                
# ======================================================================

class Ring(Curve):
    def __init__(self, radius=0.5, thickness=None, num_circ_points=36, *args, **kwargs):
        self._ring_radius = radius
        if thickness is None:
            self._thickness = 0.2 * self._ring_radius
        else:
            self._thickness = thickness
        self._num_circ_points = int(num_circ_points)

        self.calculate_ring_points()

        super().__init__(radius=0.01, points=self._ring_points, *args, **kwargs)

        self.thickness = self._thickness
        
    def calculate_ring_points(self):
        dphi = 2.*math.pi / self._num_circ_points
        phi = numpy.arange(self._num_circ_points + 1)*dphi
        self._ring_points = numpy.empty( [self._num_circ_points + 1, 3], dtype=numpy.float32 )
        self._ring_points[:, 0] = 0.
        self._ring_points[:, 1] = self._ring_radius * numpy.sin(phi)
        self._ring_points[:, 2] = self._ring_radius * numpy.cos(phi)
        # Make real sure the last point is the first point
        self._ring_points[-1, :] = self._ring_points[0, :]
        
    @property
    def radius(self):
        self._ring_radius

    @radius.setter
    def radius(self, val):
        self._ring_radius = val
        self.calculate_ring_points()
        self.points = self._ring_points

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        # This is what I call "radius" in the Curve class
        Curve.radius.fset(self, self._thickness)
        # # I want to update the first and last transverse,
        # #   so I'm gonna screw around with FixedLengthCurve's
        # #   internal data.  This is really ugly.  I should
        # #   have something like a "closed" parameter
        # #   for FixedLengthCurve
        # axis1 = self._points[-2, :] - self._points[-1, :]
        # axis2 = self._points[0, :] - self._points[1, :]
        # self._transverse[0, :] = axis2-axis1
        # self._transverse[0, :] *= self._radius / math.sqrt(numpy.square(self._transverse[0, :]).sum())
        # self._transverse[-1, :] = self._transverse[0, :]
        self.broadcast("update vertices")

    # Override the version in FixedLength Curve for this special case
    def make_transverse(self):
        num = self.numpoints
        axes = numpy.empty( ( num, 3 ) )
        axes[:-1, :] = self._points[1:num, :] - self._points[:num-1, :]
        axes[-1, :] = axes[0, :]
        hatxes = axes / numpy.sqrt( numpy.square(axes).sum(axis=1) )[:, numpy.newaxis]
        self._transverse = numpy.empty( self._points.shape )
        self._transverse[1:num, :] = axes[1:, :] - axes[:-1, :]
        self._transverse[0, :] = self._transverse[num-1, :]
        transmag = numpy.sqrt( numpy.square(self._transverse[:num, :]).sum(axis=1) )
        self._transverse[:num, :] *= self._radius / transmag[:, numpy.newaxis]
        
# ======================================================================

class Helix(Curve):
    """A helix (spring), rendered as a tube around a helical path.

    Initially oriented along the x-axis, with the first point at +z, and
    winding left-handed.  The object's position is at one end of the
    center line of the spring.
    """

    def __init__(self, radius=1., coils=5., length=1., thickness=None,
                 num_circ_points=12, *args, **kwargs):
        """Parameters:
        
        radius — The radius of the whole spring
        coils — The number of coils in the spring (can be fractional, but you'll get an approximation)
        length — The length of the spring (same as scale[0])
        num_circ_points — The number of points on the path in one winding of the spring (default: 12)
        thickness — The thickness of the actual spring wire (default: 0.05 * radius)

        Plus the usual GrObject paramters.
        """
        
        self._helixradius = radius
        if thickness is None:
            self._thickness = 0.05 * self._helixradius
        else:
            self._thickness = thickness
        self._coils = coils
        self._num_circ_points = int(num_circ_points)

        self._ncenters = int(math.floor(coils * num_circ_points + 0.5)) + 1

        self.calculate_helix_points()

        super().__init__(radius=self._thickness, points=self._helix_points, *args, **kwargs)
        
    def calculate_helix_points(self):
        dphi = 2.*math.pi / self._num_circ_points
        dx = 1. / (self._ncenters - 1)

        centcounter = numpy.arange(self._ncenters)
        self._helix_points = numpy.empty( [ self._ncenters, 3], dtype=numpy.float32 )
        self._helix_points[:, 0] = dx * centcounter
        self._helix_points[:, 1] = self._helixradius * numpy.sin(dphi * centcounter)
        self._helix_points[:, 2] = self._helixradius * numpy.cos(dphi * centcounter)
        
    @property
    def length(self):
        """The length of the spring (same as magnitude of axis)."""
        self._axis.mag

    @length.setter
    def length(self, value):
        self.axis *= value/self._axis.mag

    @property
    def radius(self):
        """The radius of the coils of the spring."""
        return self._helixradius

    @radius.setter
    def radius(self, value):
        self._helixradius = value
        self.calculate_helix_points()
        self.points = self._helix_points

    @property
    def thickness(self):
        """The thickness of the actual spring wire."""
        return self._thickness
    
    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        # This is what I call "radius" in the Curve class
        Curve.radius.fset(self, self._thickness)
            
# ======================================================================

def main():
    doaxes = False
    dobox1 = False
    dobox2 = False
    doplatonics = False
    doball = True
    dostaticball = True
    dopeg = False
    dopeg2 = False
    doblob = True
    doarrow = False
    dohelix = False
    docurve = False
    dosincurve = False
    dohairpin = False
    dobigcurve = False
    doring = False
    domanyelongatedboxes = False

    # Make objects
    sys.stderr.write("Making boxes and peg and other things.\n")

    if doaxes:
        xax = Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(1, 0, 0), color=color.red)
        yax = Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(0, 1, 0), color=color.green)
        zax = Arrow(pos=(0, 0, 0), shaftwidth=0.05, headwidth=0.125, headlength=0.2, fixedwidth=True,
                    axis=(0, 0, 1), color=color.blue)
    if dobox1:
        sys.stderr.write("Making box1.\n")
        box1 = Box(pos=(-0.5, -0.5, 0), length=0.25, width=0.25, height=0.25, color=[0.5, 0., 1.])

    if dobox2:
        sys.stderr.write("Making box2.\n")
        box2_base_trail_radius = 0.05
        box2 = Box(pos=( 0.5,  0.5, 0), length=0.25, width=0.25, height=0.25, color=color.red,
                   trail_radius=box2_base_trail_radius)

    if doplatonics:
        sys.stderr.write("Making platonic solids.\n")
        r = 2.0
        posphis = [ 0., 2./5*math.pi, 4./5.*math.pi, 6./5.*math.pi, 8./5.*math.pi ]
        colors = [ color.red, color.green, color.blue, color.yellow, color.magenta ]
        tet = Tetrahedron( pos=( r*math.cos(posphis[0]), r*math.sin(posphis[0]), 0.), color=colors[0] )
        cube = Box( pos=( r*math.cos(posphis[1]), r*math.sin(posphis[1]), 0.), color=colors[1] )
        octa = Octahedron( pos=( r*math.cos(posphis[2]), r*math.sin(posphis[2]), 0.), color=colors[2] )
        dod = Dodecahedron( pos=( r*math.cos(posphis[3]), r*math.sin(posphis[3]), 0.), color=colors[3] )
        ico = Icosahedron( pos=( r*math.cos(posphis[4]), r*math.sin(posphis[4]), 0.), color=colors[4] )
        
    if dopeg:
        sys.stderr.write("Making peg.\n")
        peg = Cylinder(pos=(0., 0., 0.), radius=0.125, color=color.orange, num_edge_points=32)
        peg.axis = (0.5, 0.5, 0.5)
        sys.stderr.write("Peg axis = {}\n".format(peg.axis))
    if dopeg2:
        sys.stderr.write("Making peg2.\n")
        peg2 = Cylinder(pos=(0., -0.25, 0.), radius=0.125, color=color.cyan, num_edge_points=6,
                        axis=(-0.5, 0.5, 0.5))
    if doblob:
        sys.stderr.write("Making blob.\n")
        blob = Ellipsoid(pos=(0., 0., 0.), length=0.5, width=0.25, height=0.125, color=color.magenta)
        blob.axis = (-0.5, -0.5, 0.5)
    if doarrow:
        sys.stderr.write("Making arrow.\n")
        arrow = Arrow(pos=(0., 0., 0.5), shaftwidth=0.05, headwidth = 0.1, headlength=0.2,
                      color=color.yellow, fixedwidth=True)
    
    if doball:
        sys.stderr.write("Making ball.\n")
        ball = Sphere(pos= (2., 0., 0.), radius=0.5, color=color.green)
        # ball = Icosahedron(pos = (2., 0., 0.), radius=0.5, color=color.green, flat=True, subdivisions=1)

    if dostaticball:
        sys.stderr.write("Making static ball.\n")
        staticball = Sphere(pos= (-1.5, -1., 1.), radius=0.5, color=[1.0, 0.7, 0.2])

    if docurve:
        sys.stderr.write("Making curve.\n")
        curvepoints = numpy.empty( [100, 3] )
        for i in range(100):
            phi = 6*math.pi * i / 50.
            curvepoints[i] = [ 0.375*math.cos(phi), 0.375*math.sin(phi), 1.5 * i*i / 5000. ]
            curvepointssofar = 1
            nextcurvepointadd = 0.
            curvepointaddevery = 0.1
            curve = Curve(radius = 0.05, color = (0.75, 1.0, 0.), points = curvepoints[numpy.newaxis, 0])
        
    if dohelix:
        sys.stderr.write("Making helix.\n")
        helix = Helix(color = (0.5, 0.5, 0.), radius=0.2, thickness=0.05, length=2., coils=5,
                      num_circ_points=12)

    if dosincurve:
        sys.stderr.write("Making sin curve.\n")
        tilt = math.pi/6.
        xvals = numpy.arange(80) / 10. - 4.
        yvals = numpy.sin( 2*math.pi * (xvals/2.) )
        zvals = yvals*numpy.sin(tilt)
        yvals = yvals*numpy.cos(tilt)
        points = numpy.empty( (len(xvals), 3) )
        points[:, 0] = xvals
        points[:, 1] = yvals
        points[:, 2] = zvals
        sincurve = Curve(radius = 0.1, color = (0.9, 0.5, 1.0), points = points)

    if dohairpin:
        # Test to make sure curves with no bends work
        sys.stderr.write("Making hairpon\n")
        tilt = 0.
        points = []
        for i in range(5):
            points.append( [ i-2, 1., 0. ] )
        numincircle = 16
        for phi in numpy.arange(numincircle) * math.pi/numincircle:
            points.append( [ points[4][0] + math.sin(phi), math.cos(phi), 0. ] )
        for i in range(5):
            points.append( [ points[4][0] - i, -1., 0. ] )
        points = numpy.array(points)
        temp = points[:, 1] * math.sin(tilt)
        points[:, 1] *= math.cos(tilt)
        points[:, 2] = temp
        hairpin = Curve( radius = 0.2, color = (0.8, 1.0, 0.2), points=points)
                           
        
    if dobigcurve:
        sys.stderr.write("Making big curve.\n")
        tilt = 0.
        xvals = 2.0 * numpy.cos( [ 0., math.pi/4., math.pi/2, 3*math.pi/4, math.pi ] )
        yvals = 2.0 * numpy.sin( [ 0., math.pi/4., math.pi/2, 3*math.pi/4, math.pi ] )
        zvals = yvals*numpy.sin(tilt)
        yvals = yvals*numpy.cos(tilt)
        points = numpy.empty( (len(xvals), 3) )
        points[:, 0] = xvals
        points[:, 1] = yvals
        points[:, 2] = zvals
        bigcurve = Curve(radius = 0.25, color = (0.75, 0.75, 0.75), points = points)
        
    if doring:
        sys.stderr.write("Making ring.\n")
        ring = Ring(color = (0.3, 0.7, 0.9), radius=0.75, thickness=0.1)
        
    if domanyelongatedboxes:
        n = 10
        sys.stderr.write("Making {} elongated boxes.\n".format(n*n))
        boxes = []
        phases = []
        for i in range(n):
            for j in range (n):
                x = i*4./n - 2.
                y = j*4./n - 2.
                phases.append(random.random()*2.*math.pi)
                col = ( random.random(), random.random(), random.random() )
                boxes.append( Box(pos=(x, y, 0.), axis=(1., -1., 1.), color=col, # color=color.red,
                                  length=1.5, width=0.05, height=0.05))


    # import pdb; pdb.set_trace()
    
    # Updates

    t = 0.
    theta = math.pi/4.
    phi = 0.
    phi2 = 0.
    fps = 30
    GrContext.print_fps = True
    printfpsevery = 30
    dphi = 2*math.pi/(4.*fps)

    GrContext.get_default_instance().gl_version_info()

    lasttime = time.perf_counter()
    nextprint = printfpsevery
    first = True
    while True:

        # Animated angle
        phi += dphi
        if phi > 2.*math.pi:
            phi -= 2.*math.pi
        phi2 += dphi/3.7284317438
        if phi2 > 2.*math.pi:
            phi2 -= 2.*math.pi

        if dobox1:
            box1.color = [ 0.5, (1. + math.sin(phi))/2., (1. + math.cos(phi2))/2. ]
            box1.up = [ 0., math.cos(phi), math.sin(phi) ]
            
        if doball:
            ball.x = 2.*math.cos(phi)
            if math.sin(phi)>0.:
                ball.rotate(dphi)
            else:
                ball.rotate(-dphi)

            if phi > math.pi/2. and phi <= 3.*math.pi/2.:
                ball.visible = False
            else:
                ball.visible = True

                
        if dobox2:
            q = numpy.array( [0., 0., -math.sin(math.pi/6.), math.cos(math.pi/6.)] )
            box2.pos = quaternion_rotate(numpy.array( [ 2.*math.sin(phi2),
                                                        1.5*math.sin(phi),
                                                        1.5*math.cos(phi) ] ),
                                         q )
            # import pdb; pdb.set_trace()
            box2.trail_radius = box2_base_trail_radius * ( 1 + 0.9*math.sin(phi2) )
            
            if first:
                box2.interval = 5
                box2.retain = 50
                box2.make_trail = True
                first = False

            if phi > math.pi:
                box2.visible = False
            else:
                box2.visible = True
                
        if doarrow:
            arrow.axis = [math.cos(phi) * (1. + 0.5*math.cos(phi)),
                          math.sin(phi) * (1. + 0.5*math.cos(phi)), 0.]
        
        if dohelix:
            helix.length = 2. + math.cos(phi)

        if docurve:
            if (curvepointssofar < curvepoints.shape[0]) and t > nextcurvepointadd:
                curve.add_point(curvepoints[curvepointssofar])
                curvepointssofar += 1
                nextcurvepointadd += curvepointaddevery
            curve.radius = 0.05 + 0.04*math.sin(phi)

            if phi2 > math.pi and phi2 < 3.*math.pi/2.:
                curve.visible = False
            else:
                # sys.stderr.write("Making curve visible.\n")
                # import pdb; pdb.set_trace()
                curve.visible = True
                
        if doring:
            ring.axis = [ math.cos(phi), math.sin(phi)*math.cos(phi2), math.sin(phi)*math.sin(phi2) ]

        if domanyelongatedboxes:
            # Rotate all the elongated boxes
            for i in range(len(boxes)):
                boxes[i].axis = numpy.array( [math.sin(theta)*math.cos(phi+phases[i]),
                                             math.sin(theta)*math.sin(phi+phases[i]),
                                             math.cos(theta)] )

        rate(fps)
        t += 1./fps
        nextprint -= 1
        if nextprint <= 0 :
            nextprint = printfpsevery
            nexttime = time.perf_counter()
            sys.stderr.write("Effective main() fps = {}\n".format(printfpsevery / (nexttime - lasttime)))
            lasttime = nexttime
                             
        



# ======================================================================

if __name__ == "__main__":
    main()
#    import cProfile
#    cProfile.run("main()", "stats")

