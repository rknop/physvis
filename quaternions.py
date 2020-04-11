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

import math
import numpy

# ======================================================================
# quaternions

# https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
#
# My quaternions are [ sin(θ/2)*ux, sin(θ/2)*uy, sin(θ/2)*uz, cos(θ/2) ]
#  Representing a rotation of θ about a unit vector (ux, uy, uz)
#
# To rotate a vector v by quaterion q, do qvq¯¹, where q¯¹ can be
#  simply composed by flipping the sign of the first 3 elements of
#  q and dividing by q·q (see quaternion_rotate)
#
# If p and q are both quaternions, then their product
#  represents rotation q followed by rotation p
#
# All quaternions must be normalized, or there will be unexpected results.
#
# NOTE!  You MUST pass p and q as numpy arrays, or things might be sad

def quaternion_multiply(p, q):
    """Multiply a vector or quaternion p by a quaternion q.

    In this context, a quaternion is [ sin(θ/2)*ux, sin(θ/2)*uy,
    sin(θ/2)*uz, cos(θ/2) ] where (ux, uy, uz) is the unit vector about
    which to rotate and θ is the angle by which to rotate.

    If p is a quaternion, the returned quaternion represents rotation q followed by rotation p

    """

    if len(p) == 3:
        px, py, pz = p
        pr = 0.
    else:
        px, py, pz, pr = p
    qx, qy, qz, qr = q
    return numpy.array( [ pr*qx + px*qr + py*qz - pz*qy,
                          pr*qy - px*qz + py*qr + pz*qx,
                          pr*qz + px*qy - py*qx + pz*qr,
                          pr*qr - px*qx - py*qy - pz*qz ] , dtype=numpy.float32 )

def quaternion_rotate(p, q):
    """Rotate vector p by quaternion q.

    In this context, a quaternion is [ sin(θ/2)*ux, sin(θ/2)*uy,
    sin(θ/2)*uz, cos(θ/2) ] where (ux, uy, uz) is the unit vector about
    which to rotate and θ is the angle by which to rotate.
    """
    qinv = q.copy()
    qinv[0:3] *= -1.
    return quaternion_multiply(q, quaternion_multiply(p, qinv))[0:3]

def axistrig_rotate(v, axis, cosangle_2, sinangle_2):
    """Rotate vector v about axis by angle θ, where cosangle_2=cos(θ/2) and sinangle_2=sin(θ/2)"""
    
    mag = math.sqrt( axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2] )
    uaxis = axis/mag
    q = numpy.array( [ uaxis[0]*sinangle_2, uaxis[1]*sinangle_2, uaxis[2]*sinangle_2, cosangle_2 ] )
    return quaternion_rotate(v, q)

    
