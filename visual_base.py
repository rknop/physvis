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
import uuid
import ctypes
import itertools

import numpy
import numpy.linalg

import OpenGL.GL as GL
import OpenGL.GLUT as GLUT

_debug_shaders = False
_print_fps = False

_OBJ_TYPE_SIMPLE = 1
_OBJ_TYPE_CURVE = 2

def rate(fps):
    """Call this in the main loop of your program to have it run at most every 1/fps seconds."""
    rater = Rater.get()
    rater.rate(fps)

def exit_whole_program():
    """Call this to have the program quit next time you call rate()."""
    Rater.exit_whole_program()
    
class Rater(threading.Event):
    """A singleton class used internally to implement rate().  Get the instance with Rater.get()"""

    _instance = None
    _exit_whole_program = False

    @staticmethod
    def get():
        """Return the singleton Rater instance"""

        if Rater._instance is None:
            Rater._instance = Rater()
        return Rater._instance

    @staticmethod
    def exit_whole_program():
        """Call this to have the program quit the next time you call rate()."""
        
        Rater._exit_whole_program = True
    
    def __init__(self):
        """Never call this."""
        
        super().__init__()
        self._time_of_last_rate_call = None
        self.clear()
        
    def rate(self, fps):
        """Call this in the main loop of your program to have it run at most every 1/fps seconds."""
        
        if Rater._exit_whole_program:
            # I wonder if I should do some cleanup?  Eh.  Whatever.
            sys.exit(0)
        if self._time_of_last_rate_call is None:
            time.sleep(1./fps)
        else:
            sleeptime = self._time_of_last_rate_call + 1./fps - time.perf_counter()
            # sys.stderr.write("Sleeping for {} seconds\n".format(sleeptime))
            if sleeptime > 0:
                time.sleep(sleeptime)
        self.wait()
        self.clear()
        self._time_of_last_rate_call = time.perf_counter()

# https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
#
# My quaternions are [ sin(θ/2)*ux, sin(θ/2)*uy, sin(θ/2)*uz, cos(θ/2) ]
#  Representing a rotation of θ about a unit vector (ux, uy,u z)
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
    """Rotate vector p by quaternion q."""
    qinv = q.copy()
    qinv[0:3] *= -1.
    return quaternion_multiply(q, quaternion_multiply(p, qinv))[0:3]

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

class Subject(object):
    """Subclass this to create something from which Observers will listen for messages."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._id = uuid.uuid4()
        self.listeners = []
        # ROB!  Print warnings about unknown arguments

    def __del__(self):
        for listener in self.listeners:
            listener.receive_message("destruct", self)

    def broadcast(self, message):
        """Call this on yourself to broadcast message to all listeners."""
        with GrContext._threadlock:
            for listener in self.listeners:
                listener.receive_message(message, self)

    def add_listener(self, listener):
        """Add Observer listener to the list of things that gets messages."""
        if not listener in self.listeners:
            self.listeners.append(listener)

    def remove_listener(self, listener):
        """Remove Observer listener from the list of things that gets messages."""
        self.listeners = [x for x in self.listeners if x != listener]

class Observer(object):
    """Subclass this to be able to get messages from a Subject.

    Must implement receive_message (if you want to do anything).

    Call the add_listener() method of a Subject object, passing self as
    an argument, to start getting messages from the Subject.

    Call the remove_listener() method of a Subject object to no logner
    get messages.  You need to do this if you want yourself to be
    deleted.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ROB!  Print errors about unknown arguments

    def receive_message(self, message, subject):
        """In your subclass, implement this to get message (usually just a text string) from Subject subject."""
        pass


# ======================================================================
#
# One object collection encapsulates a set of objects that can
#  all be drawn with the same shader.


class GLObjectCollection(Observer):
    """The base class for a collection of openGL objects, used internally by a drawing context.

    It makes some assumptions about the shader that will be used with
    the object collection.  There are three Uniform buffers, with one
    element array represented by the buffer for each objectin the
    collection.
       — a model matrix uniform buffer (for transforming the object) -- mat4
       — a model normal matrix uniform buffer (for transforming the normals for light interaction) -- mat3 (**)
       — a color uniform buffer (the color of the object) -- vec4

    Objects added to a GLObjectCollection must have:
       _id — An id that is unique for all objects anywhere in the code
       visible — True or False if the object should be drawn
       _color — A 4-element float32 numpy array (r, g, b, opacity) (opacity isn't currently used)
       model_matrix — a 16-element float32 numpy array
       inverse_model_matrix — a 12-element float32 numpy array (3x3 plus std140 layout padding)
      
    """
    
    def __init__(self, context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxnumobjs = 512       # Must match the array length in the shader!!!!  Rob, do better.
        self.objects = {}
        self.object_index = {}
        self.numobjects = 0

        self.context = context

    def initglstuff(self):
        self.modelmatrixbuffer = GL.glGenBuffers(1)
        # sys.stderr.write("self.modelmatrixbuffer = {}\n".format(self.modelmatrixbuffer))
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
        # 4 bytes per float * 16 floats per object
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 4 * 16 * self.maxnumobjs, None, GL.GL_DYNAMIC_DRAW)

        self.modelnormalmatrixbuffer = GL.glGenBuffers(1)
        # sys.stderr.write("self.modelnormalmatrixbuffer = {}\n".format(self.modelnormalmatrixbuffer))
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
        # 4 bytes per float * 9 floats per object
        #  BUT!  Because of std140 layout, there's actually 12 floats per object,
        #    as the alignment of each row of the matrix is like a vec4 rather than a vec3
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 4 * 12 * self.maxnumobjs, None, GL.GL_DYNAMIC_DRAW)

        self.colorbuffer = GL.glGenBuffers(1)
        # sys.stderr.write("self.colorbuffer = {}\n".format(self.colorbuffer))
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.colorbuffer)
        # 4 bytes per float * 4 floats per object
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 4 * 4 * self.maxnumobjs, None, GL.GL_DYNAMIC_DRAW)

        dex = GL.glGetUniformBlockIndex(self.shader.progid, "ModelMatrix")
        # sys.stderr.write("ModelMatrix block index (progid={}): {}\n".format(self.shader.progid, dex))
        GL.glUniformBlockBinding(self.shader.progid, dex, 0);

        dex = GL.glGetUniformBlockIndex(self.shader.progid, "ModelNormalMatrix")
        # sys.stderr.write("ModelNormalMatrix block index (progid={}): {}\n".format(self.shader.progid, dex))
        GL.glUniformBlockBinding(self.shader.progid, dex, 1);

        dex = GL.glGetUniformBlockIndex(self.shader.progid, "Colors")
        # sys.stderr.write("Colors block index (progid={}): {}\n".format(self.shader.progid, dex))
        GL.glUniformBlockBinding(self.shader.progid, dex, 2);

        self.bind_uniform_buffers()
        
        # In the past, I was passing a model matrix for each
        # and every vertex.  That was profligate.  I'm leaving this
        # comment here, though, as it's got a pointer to docs how I made that work.
        # See https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_instanced_arrays.txt
        # and http://sol.gfxile.net/instancing.html

    def remove_object(self, obj):
        if not obj._id in self.objects:
            return

        self.context.run_glcode(lambda : self.do_remove_object(obj))

    def bind_uniform_buffers(self):
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.modelmatrixbuffer)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 1, self.modelnormalmatrixbuffer)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 2, self.colorbuffer)
        
        
    def update_object_matrix(self, obj):
        if not obj.visible: return

        if not obj._id in self.objects:
            sys.stderr.write("...object not found whose matrix was to be updated!!\n")
            return

        # sys.stderr.write("...found at {}!\n".format(i))
        # sys.stderr.write("\nmatrixdata:\n{}\n".format(obj.model_matrix))
        # sys.stderr.write("\nnormalmatrixdata:\n{}\n".format(obj.inverse_model_matrix))

        self.context.run_glcode(lambda : self.do_update_object_matrix(obj))

    def do_update_object_matrix(self, obj):
        with GrContext._threadlock:
            if not obj._id in self.objects:
                return
            dex = self.object_index[obj._id]
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
            GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*16, obj.model_matrix.flatten())
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
            GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*12, obj.inverse_model_matrix.flatten())
            self.context.update()

    def do_remove_object_uniform_buffer_data(self, obj):
        with GrContext._threadlock:
            if not obj._id in self.objects: return
            dex = self.object_index[obj._id]
            # sys.stderr.write("Removing uniform buffer data at dex={}\n".format(dex))
            if dex < len(self.objects)-1:
                GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
                data = GL.glGetBufferSubData( GL.GL_UNIFORM_BUFFER, (dex+1)*4*16, (len(self.objects)-(dex+1))*4*16 )
                GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*16, data)

                GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
                data = GL.glGetBufferSubData( GL.GL_UNIFORM_BUFFER, (dex+1)*4*12, (len(self.objects)-(dex+1))*4*12 )
                GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*12, data)

                GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.colorbuffer)
                data = GL.glGetBufferSubData( GL.GL_UNIFORM_BUFFER, (dex+1)*4*4, (len(self.objects)-(dex+1))*4*4 )
                GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*4, data)
            
    def update_object_color(self, obj):
        if not obj.visible: return
        if not obj._id in self.objects:
            return

        self.context.run_glcode(lambda : self.do_update_object_color(obj))

    def do_update_object_color(self, obj):
        with GrContext._threadlock:
            if not obj._id in self.objects:
                return
            # sys.stderr.write("Updating an object color.\n")
            dex = self.object_index[obj._id]
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.colorbuffer)
            GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*4, obj._color)
            self.context.update()
            
    def receive_message(self, message, subject):
        # sys.stderr.write("Got message \"{}\" from {}\n".format(message, subject._id))
        if message == "update color":
            self.update_object_color(subject)
        if message == "update matrix":
            self.update_object_matrix(subject)
        if message == "update vertices":
            self.update_object_vertices(subject)

# ======================================================================
# SimpleObjectCollection
#
# This is for objects that don't require a geometry shader (so no curves).
#
# Shaders take as input for each vertex of each triangle
#  location  (4 floats per vertex)
#  normal    (3 floats per vertex)
#  index     (1 index per vertex)
#
# The object collection points to a set of 3 VBOs with this information
# for each vertex of each object.  There's a single VBO so that the
# whole damn thing can be drawn in one call to OpenGL for efficiency
# purposes.  This means that I've got to do all sorts of memory
# management manually in order to keep track of which data goes with
# which object.  (I could reduce the amount of data per object by using
# EBOs, but that would also make the data management more complicated.)
#
# Shaders also have a few arrays of uniforms, one element of the array
# for each object; the input "index" is points into this arary.
#
#  model matrix  (16 floats per object)
#  model normal matrix (something like an inverse)  (12* floats per object)
#  color     (4 floats per vertex)
#
#  * really, it's a mat3, so you'd think 9 floats per object.  However,
#  The OpenGL std140 layout means that things are aligned on vec4
#  bounadries, so there's an extra "junk" float at the end of each
#  row of the matrix.
#
# I'm gonna have to remind myself why location and color need 4, not 3,
# floats.  (For location, it's so that the transformation matrices can
# be 4×4 to allow offsets as well as rotations and scales.)  (For color,
# alpha?  It's not implemented, but maybe that's what I was thinking.)
#

class SimpleObjectCollection(GLObjectCollection):
    """A collection of "simple" objects.

    Each object is represented by a number of triangles.  An object that
    goes into one of these collections must have:
      — num_triangles : the number of triangles in the object
      — vertexdata : a numpy array of float32 that has a sequence of vec4 (w=1), 3 for each triangle (w = 1)
      — normaldata : a numpy array of float32 that has a sequence of vec3, 3 for each triangle

    They must also meet the requirements of GLObjectCollection
    """
    
    def __init__(self, context, *args, **kwargs):
        super().__init__(context, *args, **kwargs)
        self.shader = Shader.get("Basic Shader", context)

        self.maxnumtris = 65536

        self.curnumtris = 0
        self.object_triangle_index = {}

        self.draw_as_lines = False

        self.is_initialized = False
        context.run_glcode(lambda : self.initglstuff())

        while not self.is_initialized:
            time.sleep(0.1)

    def initglstuff(self):
        super().initglstuff()
        
        self.vertexbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexbuffer)
        # 4 bytes per float * 4 floats per vertex * 3 vertices per triangle
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 4 * 3 * self.maxnumtris, None, GL.GL_STATIC_DRAW)

        self.normalbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normalbuffer)
        # 4 bytes per float * 3 floats per vertex * 3 vertices per triangle
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 3 * 3 * self.maxnumtris, None, GL.GL_STATIC_DRAW)

        self.objindexbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        # 4 bytes per int * 1 int per vertex * 3 vertices per triangle
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 1 * 3 * self.maxnumtris, None, GL.GL_STATIC_DRAW)
        
        self.VAO = GL.glGenVertexArrays(1)

        self.bind_vertex_attribs()
        self.is_initialized = True

    def bind_vertex_attribs(self):
        GL.glBindVertexArray(self.VAO)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexbuffer)
        GL.glVertexAttribPointer(0, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normalbuffer)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        GL.glVertexAttribIPointer(2, 1, GL.GL_INT, 0, None)
        GL.glEnableVertexAttribArray(2)

    def add_object(self, obj):
        # Make sure not to double-add
        if obj._id in self.objects:
            return

        if len(self.objects) >= self.maxnumobjs:
            raise Exception("Error, I can currently only handle {} objects.".format(self.maxnumobjs))
        if self.curnumtris + obj.num_triangles > self.maxnumtris:
            raise Exception("Error, I can currently only handle {} triangles.".format(self.maxnumtris))

        self.context.run_glcode(lambda : self.do_add_object(obj))

    def do_add_object(self, obj):
        with GrContext._threadlock:
            if obj._id in self.objects:
                return
            self.object_triangle_index[obj._id] = self.curnumtris
            self.objects[obj._id] = obj
            self.curnumtris += obj.num_triangles
            self.object_index[obj._id] = len(self.objects) - 1
            self.push_all_object_info(obj)
            obj.add_listener(self)
            # sys.stderr.write("Up to {} objects, {} triangles.\n".format(len(self.objects), self.curnumtris))

    def do_remove_object(self, obj):
        with GrContext._threadlock:
            if not obj._id in self.objects:
                return
            dex = self.object_index[obj._id]
            # sys.stderr.write("Removing object at dex={} out of {}\n".format(dex, len(self.objects)))
            if dex < len(self.objects)-1:
                srcoffset = self.object_triangle_index[obj._id] + obj.num_triangles
                dstoffset = self.object_triangle_index[obj._id]
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexbuffer)
                data = GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, srcoffset*4*4*3, (self.curnumtris - srcoffset)*4*4*3 )
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dstoffset*4*4*3, data)

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normalbuffer)
                data = GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, srcoffset*4*3*3, (self.curnumtris - srcoffset)*4*3*3 )
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dstoffset*4*3*3, data)

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
                data = numpy.empty( (self.curnumtris - srcoffset)*3, dtype=numpy.int32 )
                GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, srcoffset*4*1*3, (self.curnumtris - srcoffset)*4*1*3,
                                    ctypes.c_void_p(data.__array_interface__['data'][0]) )
                data[:] -= 1
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dstoffset*4*1*3, data)

                self.do_remove_object_uniform_buffer_data(obj)

            for objid in self.objects:
                if self.object_index[objid] > dex:
                    self.object_triangle_index[objid] -= obj.num_triangles
                    self.object_index[objid] -= 1
            self.curnumtris -= obj.num_triangles

            del self.objects[obj._id]
            del self.object_index[obj._id]
            del self.object_triangle_index[obj._id]
            obj.remove_listener(self)
            
            self.context.update()
                
    # Updates positions of verticies and directions of normals.
    # Can NOT change the number of vertices
    def update_object_vertices(self, obj):
        if not obj.visible: return
        if not obj._id in self.objects: return
        self.context.run_glcode(lambda : self.do_update_object_vertex(obj))

    def do_update_object_vertex(self, obj):
        with GrContext._threadlock:
            if not obj._id in self.objects:
                return
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*4*3, obj.vertexdata)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normalbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*3*3, obj.normaldata)
            self.context.update()


    def push_all_object_info(self, obj):
        if not obj._id in self.objects: return
        dex = self.object_index[obj._id]
        
        # sys.stderr.write("Pushing object info for index {} (with {} triangles, at offset {}).\n"
        #                  .format(dex, obj.num_triangles,
        #                          self.object_triangle_index[obj._id]))
        # sys.stderr.write("\nvertexdata: {}\n".format(obj.vertexdata))
        # sys.stderr.write("\nnormaldata: {}\n".format(obj.normaldata))
        # sys.stderr.write("\ncolordata: {}\n".format(obj.colordata))
        # sys.stderr.write("\nmatrixdata: {}\n".format(obj.matrixdata))
        # sys.stderr.write("\nnormalmatrixdata: {}\n".format(obj.normalmatrixdata))
        # sys.exit(20)

        # sys.stderr.write("Pushing vertexdata for obj {}\n".format(dex))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexbuffer)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*4*3, obj.vertexdata)

        # sys.stderr.write("Pushing normaldata for obj {}\n".format(dex))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normalbuffer)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*3*3, obj.normaldata)

        objindexcopies = numpy.empty(self.objects[obj._id].num_triangles*3, dtype=numpy.int32)
        objindexcopies[:] = dex
        # sys.stderr.write("Pushing object_index for obj {}\n".format(dex))
        # sys.stderr.write("objindexcopies = {}\n".format(objindexcopies))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*1*3, objindexcopies)
        
        self.do_update_object_matrix(obj)
        self.do_update_object_color(obj)

        self.context.update()    # Redundant... it just happened in the last two function calls

    # Never call this directly!  It should only be called from within the
    #   draw method of a GrContext
    def draw(self):
        with GrContext._threadlock:
            # sys.stderr.write("Drawing Simple Object Collection with shader progid {}\n".format(self.shader.progid))
            GL.glUseProgram(self.shader.progid)
            self.bind_uniform_buffers()
            self.bind_vertex_attribs()
            self.shader.set_perspective(self.context._fov, self.context.width/self.context.height,
                                        self.context._clipnear, self.context._clipfar)
            self.shader.set_camera_posrot(self.context._camx, self.context._camy, self.context._camz,
                                          self.context._camtheta, self.context._camphi)

            if self.draw_as_lines:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            else:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glBindVertexArray(self.VAO)
            # sys.stderr.write("About to draw {} triangles\n".format(self.curnumtris))
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.curnumtris*3)
            # sys.stderr.write("...done drawing triangles.")


# ======================================================================
# CurveCollection

class CurveCollection(GLObjectCollection):
    """A collection of curves defined by a sequence of points.

    ROB WRITE MORE
    """
    
    def __init__(self, context, *args, **kwargs):
        super().__init__(context, *args, **kwargs)
        self.shader = Shader.get("Curve Tube Shader", context)
        self.maxnumlines=16384

        self.curnumlines = 0
        self.line_index = {}

        self.draw_as_lines = False
        
        self.is_initialized = False
        context.run_glcode(lambda : self.initglstuff())

        while not self.is_initialized:
            time.sleep(0.1)

    def initglstuff(self):
        super().initglstuff()

        self.linebuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.linebuffer)
        # 4 bytes per float * 4 floats per vertex * 2 vertices per line
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 4 * 2 * self.maxnumlines, None, GL.GL_STATIC_DRAW)

        self.transversebuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.transversebuffer)
        # Same length
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 4 * 2 * self.maxnumlines, None, GL.GL_STATIC_DRAW)

        self.objindexbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        # 4 bytes per int * 1 int per vertex * 2 vertices per line
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4 * 1 * 2 * self.maxnumlines, None, GL.GL_STATIC_DRAW)
        
        self.VAO = GL.glGenVertexArrays(1)

        self.bind_vertex_attribs()
        self.is_initialized = True

    def bind_vertex_attribs(self):
        GL.glBindVertexArray(self.VAO)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.linebuffer)
        GL.glVertexAttribPointer(0, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.transversebuffer)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        GL.glVertexAttribIPointer(2, 1, GL.GL_INT, 0, None)
        GL.glEnableVertexAttribArray(2)

    def add_object(self, obj):
        if obj._id in self.objects:
            return

        if len(self.objects) >= self.maxnumobjs:
            raise Exception("Error, I can currently only handle {} objects.".format(self.maxnumobjs))
        if self.curnumlines + (obj.points.shape[0]-1) > self.maxnumlines:
            raise Exception("Error, I can currently only handle {} lines.".format(self.maxnumlines))

        self.context.run_glcode(lambda : self.do_add_object(obj))

    def do_add_object(self, obj):
        with GrContext._threadlock:
            self.objects[obj._id] = obj
            self.line_index[obj._id] = self.curnumlines
            obj.add_listener(self)
            self.curnumlines += obj.points.shape[0]-1
            # sys.stderr.write("Up to {} curves, {} curve segments.\n".format(len(self.objects), self.curnumlines))

            n = len(self.objects) - 1
            self.object_index[obj._id] = n
            self.push_all_object_info(obj)
        
    def do_remove_object(self, obj):
        with GrContext._threadlock:
            if not obj._id in self.objects: return
            dex = self.object_index[obj._id]
            if dex < len(self.objects)-1:
                srcoffset = self.line_index[obj._id] + (obj.points.shape[0]-1)
                dstoffset = self.line_index[obj._id]
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.linebuffer)
                data = GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, srcoffset*4*4*2, (self.curnumlines - srcoffset)*4*4*2 )
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dstoffset*4*4*2, data)

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.transversebuffer)
                data = GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, srcoffset*4*4*2, (self.curnumlines - srcoffset)*4*4*2 )
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dstoffset*4*4*2, data)

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
                data = numpy.empty( (self.curnumlines - srcoffset)*2, dtype=numpy.int32 )
                GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, srcoffset*4*1*2, (self.curnumlines - srcoffset)*4*1*2,
                                    ctypes.c_void_p(data.__array_interface__['data'][0]) )
                data[:] -= 1
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dstoffset*4*1*2, data)
                
                self.do_remove_object_uniform_buffer_data(obj)

            numlinestoyank = obj.points.shape[0]-1
            for objid in self.objects:
                if self.object_index[objid] > dex:
                    self.line_index[objid] -= numlinestoyank
                    self.object_index[objid] -= 1
            self.curnumlines -= numlinestoyank

            del self.objects[obj._id]
            del self.object_index[obj._id]
            del self.line_index[obj._id]
            obj.remove_listener(self)
            
            self.context.update()

    def update_object_vertices(self, obj):
        if not obj.visible: return
        if not obj._id in self.objects:
            return

        self.context.run_glcode(lambda : self.do_update_object_points(obj))

    def do_update_object_points(self, obj):
        with GrContext._threadlock:
            if obj.points.shape[0] == 0:
                return
            if not obj._id in self.objects:
                return
            
            linespoints = numpy.empty( [ (obj.points.shape[0]-1)*2, 4 ], dtype=numpy.float32 )
            transpoints = numpy.empty( [ (obj.trans.shape[0]-1)*2, 4 ], dtype=numpy.float32 )
            linespoints[:, 3] = 1.
            transpoints[:, 3] = 0.
            linespoints[0, 0:3] = obj.points[0, :]
            transpoints[0, 0:3] = obj.trans[0, :]
            for i in range(1, obj.points.shape[0]-1):
                linespoints[2*i - 1, 0:3] = obj.points[i, :]
                transpoints[2*i - 1, 0:3] = obj.trans[i, :]
                linespoints[2*i, 0:3] = obj.points[i, :]
                transpoints[2*i, 0:3] = obj.trans[i, :]
            linespoints[-1, 0:3] = obj.points[-1, :]
            transpoints[-1, 0:3] = obj.trans[-1, :]

            offset = self.line_index[obj._id]
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.linebuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, offset*4*4*2, linespoints)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.transversebuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, offset*4*4*2, transpoints)
            self.context.update()
        
    def push_all_object_info(self, obj):
        if not obj._id in self.objects:
            return
        
        self.do_update_object_points(obj)

        dex = self.object_index[obj._id]
        objindexcopies = numpy.empty(2*(obj.points.shape[0]-1), dtype=numpy.int32)
        objindexcopies[:] = dex
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.line_index[obj._id]*4*1*2, objindexcopies)
        
        self.do_update_object_matrix(obj)
        self.do_update_object_color(obj)

        self.context.update()
            
    # Never call this directly!  It should only be called from within the
    #   draw method of a GrContext
    #
    # (This has a lot of redundant code with the same method in SimpleObjectCollection.)
    def draw(self):
        with GrContext._threadlock:
            # sys.stderr.write("Drawing Curve Tube Collection with shader progid {}\n".format(self.shader.progid))
            GL.glUseProgram(self.shader.progid)
            self.bind_uniform_buffers()
            self.bind_vertex_attribs()
            self.shader.set_perspective(self.context._fov, self.context.width/self.context.height,
                                        self.context._clipnear, self.context._clipfar)
            self.shader.set_camera_posrot(self.context._camx, self.context._camy, self.context._camz,
                                          self.context._camtheta, self.context._camphi)

            if self.draw_as_lines:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            else:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glBindVertexArray(self.VAO)
            # sys.stderr.write("About to draw {} lines\n".format(self.curnumlines))
            GL.glDrawArrays(GL.GL_LINES, 0, self.curnumlines*2)
            # sys.stderr.write("...done drawing lines\n");
        
# ======================================================================
# Context in which we could put objects.
#

class GrContext(Observer):
    """Encapsulates a window (or widget) and OpenGL context in which to draw.

    Right now, the only safe way to get a context is to call
    GrContext.get_default_instance().  It will give you a GLUT window.
    Future plans: allow more than one context, and also allow a context
    that would be a QWidget rather than a GLUT window.
    """
    
    _threadlock = threading.RLock()

    _default_instance = None

    def get_default_instance(*args, **kwargs):
        if GrContext._default_instance is None:
            GrContext._default_instance = GLUTContext(*args, **kwargs)
        return GrContext._default_instance

    def update(self):
        """Call this to flag the OpenGL renderer that things need to be redrawn."""
        raise Exception("GrContext subclasses need to implement update().")

    def run_glcode(self, func):
        """Call this to give a function that should be run in the GUI context."""
        raise Exception("GrContext subclasses need to implement run_glcode().")
    
    def gl_version_info(self):
        self.run_glcode(lambda : GrContext.do_gl_version_info())

    # It seems to be unhappy if you call this outside
    #  of a proper OpenGL Context.  Instead call
    #  the gl_version_info method of a GrContext instance.
    @staticmethod
    def do_gl_version_info():
        sys.stderr.write("OpenGL version: {}\n".format(GL.glGetString(GL.GL_VERSION)))
        sys.stderr.write("OpenGL renderer: {}\n".format(GL.glGetString(GL.GL_RENDERER)))
        sys.stderr.write("OpenGL vendor: {}\n".format(GL.glGetString(GL.GL_VENDOR)))
        sys.stderr.write("OpenGL shading language version: {}\n"
                         .format(GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)))

# ======================================================================

class GLUTContext(GrContext):

    _class_init_1 = False
    _class_init_2 = False

    
    # ======================================================================
    # Class methods

    @staticmethod
    def class_init(object):
        # sys.stderr.write("Starting class_init\n")

        with GrContext._threadlock:
            if GLUTContext._class_init_1:
                return

            GLUT.glutInit(len(sys.argv), sys.argv)
            GLUT.glutInitContextVersion(3, 3)
            GLUT.glutInitContextFlags(GLUT.GLUT_FORWARD_COMPATIBLE)
            GLUT.glutInitContextProfile(GLUT.GLUT_CORE_PROFILE)
            GLUT.glutSetOption(GLUT.GLUT_ACTION_ON_WINDOW_CLOSE, GLUT.GLUT_ACTION_GLUTMAINLOOP_RETURNS)
            GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)

            ### res = GL.glInitSeparateShaderObjectsARB() # ROB check error return
            ### GL.glEnable(res)

            GLUTContext._class_init_1 = True

    @staticmethod
    def class_init_2(instance):
        # sys.stderr.write("Starting class_init_2\n")
        GrContext._default_instance = instance
        with GrContext._threadlock:
            if not GLUTContext._class_init_1:
                raise Exception("class_init_2() called with _class_init_1 False")

            if GLUTContext._class_init_2:
                return

            GLUTContext.idle_funcs = []
            GLUTContext.things_to_run = queue.Queue()

            # sys.stderr.write("Starting GLUT.GLUT thread...\n")
            # GrContext.thread = threading.Thread(target = lambda : GLUTContext.thread_main(instance) )
            GLUTContext.thread = threading.Thread(target=GLUTContext.thread_main, args=(instance,))
            GLUTContext.thread.daemon = True
            GLUTContext.thread.start()
            # sys.stderr.write("GrContext.thread.ident = {}\n".format(GrContext.thread.ident))
            # sys.stderr.write("Current thread ident = {}\n".format(threading.get_ident()))
            # sys.stderr.write("Main thread ident = {}\n".format(threading.main_thread().ident))

            GLUTContext._class_init_2 = True

    # There's a race condition here on idle_funcs and things_to_run
    @staticmethod
    def thread_main(instance):
        sys.stderr.write("Starting thread_main\n")
        GLUT.glutInitWindowSize(instance.width, instance.height)
        GLUT.glutInitWindowPosition(0, 0)
        instance.window = GLUT.glutCreateWindow(instance.title)
        GLUT.glutSetWindow(instance.window)
        GLUT.glutIdleFunc(lambda: GLUTContext.idle())
        sys.stderr.write("Going into GLUT.GLUT main loop.\n")
        GLUT.glutMainLoop()

    @staticmethod
    def add_idle_func(func):
        GLUTContext.idle_funcs.append(func)

    @staticmethod
    def remove_idle_func(func):
        GLUTContext.idle_funcs = [x for x in GLUTContext.idle_funcs if x != func]

    @staticmethod
    def idle():
        with GrContext._threadlock:
            try:
                while not GLUTContext.things_to_run.empty():
                    func = GLUTContext.things_to_run.get()
                    func()
            except queue.Empty:
                pass

            for func in GLUTContext.idle_funcs:
                func()


    # ======================================================================
    # Instance methods

    def __init__(self, width=500, height=400, title="GLUT.GLUT", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # sys.stderr.write("Starting __init__")
        GLUTContext.class_init(self)

        self.window_is_initialized = False
        self.width = width
        self.height = height
        self.title = title
        self.framecount = 0
        self.vtxarr = None
        self.vboarr = None
        self.colorbuffer = None

        self._camx = 0.
        self._camy = 0.
        self._camz = 5.
        self._camtheta = math.pi/2.
        self._camphi = 0.

        self._fov = math.pi/4.
        self._clipnear = 0.1
        self._clipfar = 1000.

        self._mousex0 = 0.
        self._mousey0 = 0.
        self._origtheta = 0.
        self._origphi = 0.
        self._origcamz = 0.

        self.simple_object_collections = []
        self.curve_collections = []

        GLUTContext.class_init_2(self)

        GLUTContext.things_to_run.put(lambda : self.gl_init())

        while not self.window_is_initialized:
            time.sleep(0.1)

        self.simple_object_collections.append(SimpleObjectCollection(self))
        self.curve_collections.append(CurveCollection(self))

        # sys.stderr.write("Exiting __init__\n")

    def gl_init(self):
        # sys.stderr.write("Starting gl_init\n")
        GLUT.glutSetWindow(self.window)
        GLUT.glutMouseFunc(lambda button, state, x, y : self.mouse_button_handler(button, state, x, y))
        GLUT.glutReshapeFunc(lambda width, height : self.resize2d(width, height))
        GLUT.glutDisplayFunc(lambda : self.draw())
        GLUT.glutVisibilityFunc(lambda state : self.window_visibility_handler(state))
        # Right now, the timer just prints FPS
        GLUT.glutTimerFunc(0, lambda val : self.timer(val), 0)
        GLUT.glutCloseFunc(lambda : self.cleanup())
        self.window_is_initialized = True
        # sys.stderr.write("Exiting gl_init\n")

    def update(self):
        GLUT.glutPostRedisplay()
        
    def run_glcode(self, func):
        # sys.stderr.write("Starting run_glcode\n")
        GLUTContext.things_to_run.put(func)
        
    def window_visibility_handler(self, state):
        if state != GLUT.GLUT_VISIBLE:
            return
        GLUT.glutSetWindow(self.window)
        with GrContext._threadlock:
            GrContext._full_init = True
        GLUT.glutVisibilityFunc(None)

    def mouse_button_handler(self, button, state, x, y):
        if button == GLUT.GLUT_RIGHT_BUTTON:
            GLUT.glutSetWindow(self.window)

            if state == GLUT.GLUT_UP:
                GLUT.glutMotionFunc(None)
                if self._camtheta > math.pi:
                    self._camtheta = math.pi
                if self._camtheta < 0.:
                    self._camtheta = 0.
                if self._camphi > 2.*math.pi:
                    self._camphi -= 2.*math.pi
                if self._camphi < 0.:
                    self._camphi += 2.*math.pi

            elif state == GLUT.GLUT_DOWN:
                self._mousex0 = x
                self._mousey0 = y
                self._origtheta = self._camtheta
                self._origphi = self._camphi
                GLUT.glutMotionFunc(lambda x, y : self.rmb_moved(x, y))

        if button == GLUT.GLUT_MIDDLE_BUTTON:
            GLUT.glutSetWindow(self.window)

            if state == GLUT.GLUT_UP:
                # sys.stderr.write("MMB up\n")
                GLUT.glutMotionFunc(None)

            elif state ==GLUT.GLUT_DOWN:
                # sys.stderr.write("MMB down\n")
                self._mousex0 = x
                self._mousey0 = y
                self._origcamz = self._camz
                GLUT.glutMotionFunc(lambda x, y : self.mmb_moved(x, y))

        if button == GLUT.GLUT_LEFT_BUTTON:
            GLUT.glutSetWindow(self.window)
            
            if state == GLUT.GLUT_UP:
                # sys.stderr.write("LMB up\n")
                GLUT.glutMotionFunc(None)

            if state == GLUT.GLUT_DOWN:
                # sys.stderr.write("LMB down\n")
                keys = GLUT.glutGetModifiers()
                if keys & GLUT.GLUT_ACTIVE_SHIFT:
                    self._mouseposx0 = x
                    self._mouseposy0 = y
                    self._origcamx = self._camx
                    self._origcamy = self._camy
                    GLUT.glutMotionFunc(lambda x, y : self.lmb_moved(x, y))
            
        if (state == GLUT.GLUT_UP) and ( button == 3 or button == 4):   # wheel up/down
            GLUT.glutSetWindow(self.window)

            if button == 3:
                self._camz *= 0.9
            else:
                self._camz *= 1.1
            self.update_cam_posrot_gl()


    def rmb_moved(self, x, y):
        dx = x - self._mousex0
        dy = y - self._mousey0
        self._camtheta = self._origtheta - dy * math.pi / self.height
        self._camphi = self._origphi + dx * 2.*math.pi / self.width
        self.update_cam_posrot_gl()


    def mmb_moved(self, x, y):
        dy = y - self._mousey0
        self._camz = self._origcamz * 10.**(dy/self.width)
        self.update_cam_posrot_gl()

    def lmb_moved(self, x, y):
        dx = x - self._mouseposx0
        dy = y - self._mouseposy0
        self._camx = self._origcamx - dx / 256.
        self._camy = self._origcamy + dy / 256.
        self.update_cam_posrot_gl()
        
    def receive_message(self, message, subject):
        sys.stderr.write("OMG!  Got message {} from subject {}, should do something!\n"
                         .format(message, subject))

    def cleanup(self):
        # sys.stderr.write("cleanup called in thread {}\n".format(threading.get_ident()))
        exit_whole_program()
        # I should do better than this:
        #  * actually clean up
        #  * think about multiple windows

    def timer(self, val):
        global _print_fps
        if _print_fps:
            sys.stderr.write("Display fps: {}\n".format(self.framecount/2.))
        self.framecount = 0
        GLUT.glutTimerFunc(2000, lambda val : self.timer(val), 0)

    def resize2d(self, width, height):
        # sys.stderr.write("In resize2d w/ size {} × {}\n".format(width, height))
        self.width = width
        self.height = height
        self.run_glcode(lambda : self.resize2d_gl())

    def resize2d_gl(self):
        GL.glViewport(0, 0, self.width, self.height)
        for collection in itertools.chain( self.simple_object_collections,
                                           self.curve_collections ):
            collection.shader.set_perspective(self._fov, self.width/self.height,
                                              self._clipnear, self._clipfar)

    def update_cam_posrot_gl(self):
        # sys.stderr.write("Moving camera to [{:.2f}, {:.2f}, {:.2f}], setting rotation to [{:.3f}, {:.3f}]\n"
        #                  .format(self._camx, self._camy, self._camz, self._camtheta, self._camphi))
        for collection in itertools.chain( self.simple_object_collections,
                                           self.curve_collections ):
            collection.shader.set_camera_posrot(self._camx, self._camy, self._camz, self._camtheta, self._camphi)

    def draw(self):
        GL.glClearColor(0., 0., 0., 0.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)

        with GrContext._threadlock:
            # sys.stderr.write("About to draw collections\n")
            for collection in itertools.chain( self.simple_object_collections,
                                               self.curve_collections ):
                
                collection.draw()

                err = GL.glGetError()
                if err != GL.GL_NO_ERROR:
                    sys.stderr.write("Error {} drawing: {}\n".format(err, gluErrorString(err)))
                    sys.exit(-1)

            GLUT.glutSwapBuffers()
            # sys.stderr.write("Done drawing collections.\n")

            Rater.get().set()
            
        self.framecount += 1

    def add_object(self, obj):
        # Try to figure out which collection this goes into for real
        if obj._object_type == _OBJ_TYPE_SIMPLE:
            self.simple_object_collections[0].add_object(obj)
        elif obj._object_type == _OBJ_TYPE_CURVE:
            self.curve_collections[0].add_object(obj)

    def remove_object(self, obj):
        for collection in itertools.chain( self.simple_object_collections,
                                           self.curve_collections ):
            collection.remove_object(obj)

# ======================================================================
# ======================================================================
# ======================================================================
# Shader objects.  There probably needs to be a separate Shader subclass
# for each GLObjectCollection subclass.

class Shader(object):
    """The base class for shader objects.

    Get shaders by asking for one with Shader.get().
    """

    _basic_shader = {}
    _curvetube_shader = {}

    @staticmethod
    def get(name, context):
        """Factory method for giving shader instances.

        There will only be one shader instance of a given type for each context. 

        name — The type of shader you want:
                  "Basic Shader" to render a SimpleObjectCollection (stack of triangles)
                  "Curve Tube Shader" to render a CurveCollection (round tubes around the curve)
        context — The context for the sader.
        """
        
        if name == "Basic Shader":
            with GrContext._threadlock:
                # sys.stderr.write("Asking for a BasicShader\n")
                if ( (not context in Shader._basic_shader) or
                     (Shader._basic_shader[context] == None) ):
                    # sys.stderr.write("Creating a new BasicShader\n")
                    Shader._basic_shader[context] = BasicShader(context)
            return Shader._basic_shader[context]

        elif name == "Curve Tube Shader":
            with GrContext._threadlock:
                # sys.stderr.write("Asking for a BasicShader\n")
                if ( (not context in Shader._curvetube_shader) or
                     (Shader._curvetube_shader[context] == None) ):
                    # sys.stderr.write("Creating a new CurveTubeShader\n");
                    Shader._curvetube_shader[context] = CurveTubeShader(context)
            return Shader._curvetube_shader[context]

        else:
            raise Exception("Unknown shader \"{}\"".format(name))

    def __init__(self, context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ROB!  Warn about unknown arguments
        self.context = context
        self._name = None
        self._shaders_destroyed = False
        self.vtxshdrid = None
        self.geomshdrid = None
        self.fragshdrid = None
        self.progid = None

    # This makes me feel very queasy.  A wait for another thread in
    #   a __del__ is probably just asking for circular references
    #   to trip you up.  *But*, I gotta run all my GL code in
    #   a single thread.  So... hurm.
    def __del__(self):
        sys.stderr.write("Shader __del__\n")
        self.context.run_glcode(lambda : self.destroy_shaders())
        while not self._shaders_destroyed:
            time.sleep(0.1)
        sys.stderr.write("...BasicShader __del__ completed\n")

    def destroy_shaders(self):
        sys.stderr.write("Shader destroy_shaders\n")
        err = GL.glGetError()

        GL.glUseProgram(0)

        GL.glDetachShader(self.progid, self.vtxshdrid)
        GL.glDetachShader(self.progid, self.fragshdrid)
        if self.geomshdrid is not None:
            GL.glDetachShader(self.progid, self.geomshdrid)
        
        GL.glDeleteShader(self.fragshdrid)
        GL.glDeleteShader(self.vtxshdrid)
        if self.geomshdrid is not None:
            GL.glDetachShader(self.progid, self.geomshdrid)
        
        GL.glDeleteProgram(self.progid)

        err = GL.glGetError()
        if err != GL.GL_NO_ERROR:
            sys.stderr.write("Error {} destroying shaders: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)

        self._shaders_destroyed = True

    def get_shader(self):
        return self.progid

    @staticmethod
    def perspective_matrix(fovy, aspect, near, far):
        # Math from
        # https://www.opengl.org/discussion_boards/showthread.php/197893-View-and-Perspective-matrices
        #
        # I should understand this
        #
        # aspect is width/height

        s = 1.0/math.tan(fovy/2.0)
        sx, sy = s / aspect, s
        zz = (far+near)/(near-far)
        zw = 2*far*near/(near-far)
        return numpy.matrix([[sx, 0, 0, 0],
                             [0, sy, 0, 0],
                             [0, 0, zz, zw],
                             [0, 0, -1, 0]], dtype=numpy.float32).T


    def init_lights_and_camera(self):
        # sys.stderr.write("Shader: init_lights_and_camera\n")
        loc = GL.glGetUniformLocation(self.progid, "ambientcolor")
        GL.glUniform3fv(loc, 1, numpy.array([0.2, 0.2, 0.2]))
        loc = GL.glGetUniformLocation(self.progid, "light1color")
        GL.glUniform3fv(loc, 1, numpy.array([0.8, 0.8, 0.8]))
        loc = GL.glGetUniformLocation(self.progid, "light1dir")
        GL.glUniform3fv(loc, 1, numpy.array([0.22, 0.44, 0.88]))
        loc = GL.glGetUniformLocation(self.progid, "light2color")
        GL.glUniform3fv(loc, 1, numpy.array([0.3, 0.3, 0.3]))
        loc = GL.glGetUniformLocation(self.progid, "light2dir")
        GL.glUniform3fv(loc, 1, numpy.array([-0.88, -0.22, -0.44]))

        self.set_perspective(self.context._fov, self.context.width/self.context.height,
                             self.context._clipnear, self.context._clipfar)
        self.set_camera_posrot(self.context._camx, self.context._camy, self.context._camz,
                               self.context._camtheta, self.context._camphi)

    def set_perspective(self, fov, aspect, near, far):
        # sys.stderr.write("Shader: set_perspective\n")
        matrix = self.perspective_matrix(fov, aspect, near,far)
        # sys.stderr.write("Perspective matrix:\n{}\n".format(matrix))
        GL.glUseProgram(self.progid)
        projection_location = GL.glGetUniformLocation(self.progid, "projection")
        GL.glUniformMatrix4fv(projection_location, 1, GL.GL_FALSE, matrix)
        self.context.update()
        
    def set_camera_posrot(self, x, y, z, theta, phi):
        # sys.stderr.write("Shader: set_camera_posrot\n")
        theta -= math.pi/2.
        if (theta >  math.pi/2): theta =  math.pi/2.
        if (theta < -math.pi/2): theta = -math.pi/2.
        if (phi > 2.*math.pi): phi -= 2.*math.pi
        if (phi < 0.): phi += 2.*math.pi
        ct = math.cos(theta)
        st = math.sin(theta)
        cp = math.cos(phi)
        sp = math.sin(phi)
        matrix = numpy.matrix([[    cp   ,   0.  ,   sp  ,  0. ],
                               [ -sp*st  ,  ct   , cp*st ,  0. ],
                               [ -sp*ct  , -st   , cp*ct ,  -z ],
                               [    0.   ,   0.  ,   0.  ,  1. ]], dtype=numpy.float32)
        # sys.stderr.write("Viewrot matrix:\n{}\n".format(matrix.T))
        GL.glUseProgram(self.progid)
        viewrot_location = GL.glGetUniformLocation(self.progid, "viewrot")
        GL.glUniformMatrix4fv(viewrot_location, 1, GL.GL_FALSE, matrix.T)
        matrix = numpy.matrix([[    1.   ,   0.  ,   0.  , -x  ],
                               [    0.   ,   1.  ,   0.  , -y  ],
                               [    0.   ,   0.  ,   1.  ,  0.  ],
                               [    0.   ,   0.  ,   0.  ,  1. ]], dtype=numpy.float32)
        # sys.stderr.write("Viewshift matrix:\n{}\n".format(matrix.T))
        viewshift_location = GL.glGetUniformLocation(self.progid, "viewshift")
        GL.glUniformMatrix4fv(viewshift_location, 1, GL.GL_FALSE, matrix.T)
        self.context.update()

# ======================================================================
 # This shader goes with _OBJ_TYPE_SIMPLE and SimpleObjectCollection

class BasicShader(Shader):
    """Shader class for SimpleObjectCollection.  (Render lots of triangles.)"""
    
    def __init__(self, context, *args, **kwargs):
        # sys.stderr.write("Initializing a Basic Shader...\n")
        super().__init__(context, *args, **kwargs)
        self._name = "Basic Shader"
        self.context.run_glcode(lambda : self.create_shaders())

    def create_shaders(self):
        err = GL.glGetError()

        vertex_shader = """
#version 330

uniform mat4 viewshift;
uniform mat4 viewrot;
uniform mat4 projection;

layout (std140) uniform ModelMatrix
{
   mat4 model_matrix[512];
};

layout (std140) uniform ModelNormalMatrix
{
   mat3 model_normal_matrix[512];
};

layout (std140) uniform Colors
{
   vec4 color[512];
};

layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Normal;
layout(location=2) in int in_Index;
out vec3 aNormal;
out vec4 aColor;

void main(void)
{
  gl_Position =  projection * viewrot * viewshift * model_matrix[in_Index] * in_Position;
  aNormal = model_normal_matrix[in_Index] * in_Normal;
  aColor = color[in_Index];
}"""

        fragment_shader = """
#version 330

uniform vec3 ambientcolor;
uniform vec3 light1color;
uniform vec3 light1dir;
uniform vec3 light2color;
uniform vec3 light2dir;

in vec3 aNormal;
in vec4 aColor;
out vec4 out_Color;

void main(void)
{
  vec3 norm = normalize(aNormal);
  vec3 diff1 = max(dot(norm, light1dir), 0.) * light1color;
  vec3 diff2 = max(dot(norm, light2dir), 0.) * light2color;
  vec3 col = (ambientcolor + diff1 + diff2) * vec3(aColor);
  out_Color = vec4(col, aColor[3]);
}"""

        if _debug_shaders: sys.stderr.write("\nAbout to compile shaders....\n")

        self.vtxshdrid = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(self.vtxshdrid, vertex_shader)
        GL.glCompileShader(self.vtxshdrid)

        if _debug_shaders: sys.stderr.write("{}\n".format(GL.glGetShaderInfoLog(self.vtxshdrid)))

        self.fragshdrid = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(self.fragshdrid, fragment_shader)
        GL.glCompileShader(self.fragshdrid)

        if _debug_shaders: sys.stderr.write("{}\n".format(GL.glGetShaderInfoLog(self.fragshdrid)))
        
        self.progid = GL.glCreateProgram()
        GL.glAttachShader(self.progid, self.vtxshdrid)
        GL.glAttachShader(self.progid, self.fragshdrid)
        GL.glLinkProgram(self.progid)

        if GL.glGetProgramiv(self.progid, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            sys.stderr.write("{}\n".format(GL.glGetProgramInfoLog(self.progid)))
            sys.exit(-1)

        GL.glUseProgram(self.progid)

        if _debug_shaders: sys.stderr.write("Basic Shader created with progid {}\n".format(self.progid))
        
        err = GL.glGetError()
        if err != GL.GL_NO_ERROR:
            sys.stderr.write("Error {} creating shaders: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)

        self.init_lights_and_camera()
            

# ======================================================================
# This goes with _OBJ_TYPE_CURVE and CurveCollection

class CurveTubeShader(Shader):
    """Shader class for CurveCollection.  Renders the object as a round tube around a given curve."""
    
    def __init__(self, context, *args, **kwargs):
        super().__init__(context, *args, **kwargs)
        # sys.stderr.write("Initializing a CurveTubeShader")
        self._name = "Curve Tube Shader"

        self.context.run_glcode(lambda : self.create_shaders())

    def create_shaders(self):
        err = GL.glGetError()
      
        vertex_shader = """
#version 330

uniform mat4 viewshift;
uniform mat4 viewrot;
uniform mat4 projection;

layout (std140) uniform ModelMatrix
{
   mat4 model_matrix[512];
};

layout (std140) uniform ModelNormalMatrix
{
   mat3 model_normal_matrix[512];
};

layout (std140) uniform Colors
{
   vec4 color[512];
};

layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Transverse;
layout(location=2) in int in_Index;
out vec3 aTransverse;
out vec4 aColor;

void main(void)
{
  gl_Position =  model_matrix[in_Index] * in_Position;
  // aTransverse = model_normal_matrix[in_Index] * in_Transverse;
  vec4 tmp = vec4(in_Transverse, 0);
  tmp = model_matrix[in_Index] * tmp;
  aTransverse = tmp.xyz;
  aColor = color[in_Index];
}"""

        skeleton_geometry_shader = """
#version 330
layout(lines) in;
in vec3 aTransverse[];
in vec4 aColor[];

layout(line_strip, max_vertices = 4) out;
out vec3 aNormal;
out vec4 bColor;

void main(void)
{
    gl_Position = gl_in[0].gl_Position + vec4(aTransverse[0], 0);
    aNormal = vec3(1, 0, 0);
    bColor = aColor[0];
    EmitVertex();

    gl_Position = gl_in[0].gl_Position;
    aNormal = vec3(1, 0, 0);
    bColor = aColor[0];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position;
    aNormal = vec3(1, 0, 0);
    bColor = aColor[1];
    EmitVertex();

    gl_Position = gl_in[1].gl_Position + vec4(aTransverse[1], 0);
    aNormal = vec3(1, 0, 0);
    bColor = aColor[1];
    EmitVertex();

    EndPrimitive();
}
"""

        geometry_shader = """
#version 330

const float PI = 3.14159265359;

uniform mat4 viewshift;
uniform mat4 viewrot;
uniform mat4 projection;

layout(lines) in;
in vec3 aTransverse[];
in vec4 aColor[];

layout(triangle_strip, max_vertices = 18) out;
out vec3 aNormal;
out vec4 bColor;

void main(void)
{
    vec4 bottompoints[8];
    vec3 bottomnormal[8];
    vec4 toppoints[8];
    vec3 topnormal[8];
    vec3 perp;
    vec3 axishat;
    vec3 transhat;
    vec4 q;
    vec4 qinv;
    vec4 tmp;
    float phi;

    axishat = vec3(gl_in[1].gl_Position - gl_in[0].gl_Position);
    axishat /= length(axishat);

    transhat = aTransverse[0] / length(aTransverse[0]);
    perp = axishat - transhat * dot(axishat, transhat);
    perp /= length(perp);

    for (int i = 0 ; i < 8 ; ++i)
    {
        phi = 2.*PI * i / 8.;
        q  = vec4( perp * sin(phi/2.), cos(phi/2.) );
        qinv = vec4( -q.xyz, q.w );

        tmp = vec4(  aTransverse[0].x * qinv.w + aTransverse[0].y * qinv.z - aTransverse[0].z * qinv.y,
                    -aTransverse[0].x * qinv.z + aTransverse[0].y * qinv.w + aTransverse[0].z * qinv.x,
                     aTransverse[0].x * qinv.y - aTransverse[0].y * qinv.x + aTransverse[0].z * qinv.w,
                    -aTransverse[0].x * qinv.x - aTransverse[0].y * qinv.y - aTransverse[0].z * qinv.z );
        
        tmp = vec4( q.w * tmp.x + q.x * tmp.w + q.y * tmp.z - q.z * tmp.y,
                    q.w * tmp.y - q.x * tmp.z + q.y * tmp.w + q.z * tmp.x,
                    q.w * tmp.z + q.x * tmp.y - q.y * tmp.x + q.z * tmp.w,
                    q.w * tmp.w - q.x * tmp.x - q.y * tmp.y - q.z * tmp.z );

        bottompoints[i] = gl_in[0].gl_Position + tmp;
        bottomnormal[i] = tmp.xyz / length(tmp.xyz);
    }

    transhat = aTransverse[1] / length(aTransverse[1]);
    perp = axishat - transhat * dot(axishat, transhat);
    perp /= length(perp);

    for (int i = 0 ; i < 8 ; ++i)
    {
        phi = 2.*PI * i / 8.;
        q  = vec4( perp * sin(phi/2.), cos(phi/2.) );
        qinv = vec4( -q.xyz, q.w );

        tmp = vec4(  aTransverse[1].x * qinv.w + aTransverse[1].y * qinv.z - aTransverse[1].z * qinv.y,
                    -aTransverse[1].x * qinv.z + aTransverse[1].y * qinv.w + aTransverse[1].z * qinv.x,
                     aTransverse[1].x * qinv.y - aTransverse[1].y * qinv.x + aTransverse[1].z * qinv.w,
                    -aTransverse[1].x * qinv.x - aTransverse[1].y * qinv.y - aTransverse[1].z * qinv.z );
        
        tmp = vec4( q.w * tmp.x + q.x * tmp.w + q.y * tmp.z - q.z * tmp.y,
                    q.w * tmp.y - q.x * tmp.z + q.y * tmp.w + q.z * tmp.x,
                    q.w * tmp.z + q.x * tmp.y - q.y * tmp.x + q.z * tmp.w,
                    q.w * tmp.w - q.x * tmp.x - q.y * tmp.y - q.z * tmp.z );

        toppoints[i] = gl_in[1].gl_Position + tmp;
        topnormal[i] = tmp.xyz / length(tmp.xyz);
    }

    gl_Position = projection * viewrot * viewshift * toppoints[7];
    bColor = aColor[1];
    aNormal = topnormal[7];
    EmitVertex();

    for (int i = 0 ; i < 8 ; ++i)
    {
        gl_Position = projection * viewrot * viewshift * toppoints[i];
        bColor = aColor[1];
        aNormal = topnormal[i];
        EmitVertex();
        gl_Position = projection * viewrot * viewshift * bottompoints[i];
        bColor = aColor[0];
        aNormal = bottomnormal[i];
        EmitVertex();
    }

    gl_Position = projection * viewrot * viewshift * bottompoints[0];
    bColor = aColor[0];
    aNormal = bottomnormal[0];
    EmitVertex();

    EndPrimitive();
}
"""
   
        fragment_shader = """
#version 330

uniform vec3 ambientcolor;
uniform vec3 light1color;
uniform vec3 light1dir;
uniform vec3 light2color;
uniform vec3 light2dir;

in vec3 aNormal;
in vec4 bColor;
out vec4 out_Color;

void main(void)
{
  vec3 norm = normalize(aNormal);
  vec3 diff1 = max(dot(norm, light1dir), 0.) * light1color;
  vec3 diff2 = max(dot(norm, light2dir), 0.) * light2color;
  vec3 col = (ambientcolor + diff1 + diff2) * vec3(bColor);
  out_Color = vec4(col, bColor[3]);
}"""

        if _debug_shaders: sys.stderr.write("\nAbout to compile vertex shader....\n")
        self.vtxshdrid = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(self.vtxshdrid, vertex_shader)
        GL.glCompileShader(self.vtxshdrid)
        if _debug_shaders: sys.stderr.write("{}\n".format(GL.glGetShaderInfoLog(self.vtxshdrid)))

        if _debug_shaders: sys.stderr.write("\nAbout to compile geometry shader....\n")
        self.geomshdrid = GL.glCreateShader(GL.GL_GEOMETRY_SHADER)
        GL.glShaderSource(self.geomshdrid, geometry_shader)
        # GL.glShaderSource(self.geomshdrid, skeleton_geometry_shader)
        GL.glCompileShader(self.geomshdrid)
        if _debug_shaders: sys.stderr.write("{}\n".format(GL.glGetShaderInfoLog(self.geomshdrid)))

        if _debug_shaders: sys.stderr.write("\nAbout to compile fragment shader....\n")
        self.fragshdrid = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(self.fragshdrid, fragment_shader)
        GL.glCompileShader(self.fragshdrid)
        if _debug_shaders: sys.stderr.write("{}\n".format(GL.glGetShaderInfoLog(self.fragshdrid)))

        if _debug_shaders: sys.stderr.write("About to create shader program...\n")
        self.progid = GL.glCreateProgram()
        GL.glAttachShader(self.progid, self.vtxshdrid)
        GL.glAttachShader(self.progid, self.geomshdrid)
        GL.glAttachShader(self.progid, self.fragshdrid)
        GL.glLinkProgram(self.progid)
        if _debug_shaders: sys.stderr.write("Shader program linked.\n")

        if GL.glGetProgramiv(self.progid, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            sys.stderr.write("{}\n".format(GL.glGetProgramInfoLog(self.progid)))
            sys.exit(-1)

        GL.glUseProgram(self.progid)

        if _debug_shaders: sys.stderr.write("Curve Tube Shader created with progid {}\n".format(self.progid))

        err = GL.glGetError()
        if err != GL.GL_NO_ERROR:
            sys.stderr.write("Error {} creating shaders: {}\n".format(err, gluErrorString(err)))
            sys.exit(-1)

        self.init_lights_and_camera()

        
# ======================================================================
# ======================================================================
# ======================================================================

class GrObject(Subject):
    """Base class for all graphical objects (Box, Sphere, etc.)"""
    
    def __init__(self, context=None, pos=None, axis=None, up=None, scale=None,
                 color=None, opacity=None, make_trail=False, interval=10, retain=50,
                 *args, **kwargs):
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
        """

        super().__init__(*args, **kwargs)

        self._object_type = _OBJ_TYPE_SIMPLE
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
            self._color = numpy.array( [1., 1., 1., 1.], dtype=numpy.float32 )
        elif color is None:
            self._color = numpy.array( [1., 1., 1., opacity], dtype=numpy.float32 )
        else:
            self._color = numpy.empty(4, dtype=numpy.float32)
            self._color[0:3] = numpy.array(color)[0:3]
            if opacity is None:
                self._color[3] = 1.
            else:
                self._color[3] = opacity

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
        if axis is not None:
            self.axis = vector(axis)

        if up is not None:
            self.up = numpy.array(up)

        self._interval = interval
        self._nexttrail = interval
        self._retain = retain
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
        # return quaternion_rotate([self._scale[0], 0., 0.], self._rotation)
        # v = numpy.array([self._scale[0], 0., 0.], dtype=numpy.float32)
        # qinv = q.copy()
        # qinv[0:3] *= -1.
        # qinv /= (q*q).sum()
        # return quaternion_multiply(q, quaternion_multiply(v, qinv))[0:3]

    # I'm not completely happy with this, because if you rotate an
    # object a lot small errors can build up.
    #
    # I figure out the rotation by building on top of the exiting
    # rotation, rather than just purely from the axis (and figuring out
    # the rotation to get to the axis from [1, 0, 0], so that the
    # object's orientation doesn't suddenly change if you make a small
    # rotation that would go past a (θ, φ) coordinate singularity.
    #
    # (Really, I should figure out how I want to deal with "up".)
    @axis.setter
    def axis(self, value):
        magaxis = numpy.sqrt(numpy.square(self._axis).sum())
        newaxis = vector(value)
        magnewaxis = numpy.sqrt(numpy.square(newaxis).sum())
        cosang = (self._axis * newaxis).sum() / ( magaxis * magnewaxis )
        if math.fabs(1.-cosang) > 0.:
            rotaxis = numpy.cross(self._axis, newaxis)
            rotaxis /= math.sqrt(numpy.square(rotaxis).sum())
            cosang_2 = math.sqrt((1+cosang)/2.)
            sinang_2 = math.sqrt((1-cosang)/2.)
            q = numpy.empty(4)
            q[0:3] = sinang_2 * rotaxis
            q[3] = cosang_2

            self._rotation = quaternion_multiply(q, self._rotation)
        self._axis = newaxis
        self._scale[0] = magnewaxis
        self.update_model_matrix()
        
        # R = math.sqrt(value[0]*value[0] + value[2]*value[2])
        # theta = math.atan2(value[1], R)
        # phi = -math.atan2(value[2], value[0])
        # q1 = numpy.array([ 0., 0., numpy.sin(theta/2.), numpy.cos(theta/2.)])
        # q2 = numpy.array([ 0., numpy.sin(phi/2.), 0., numpy.cos(phi/2.)])
        # self._rotation = quaternion_multiply(q2, q1)
        # self._scale[0] = math.sqrt( value[0]*value[0] + value[1]*value[1] + value[2]*value[2] )
        # self.update_model_matrix()

    @property
    def up(self):
        """Not implemendted!!!!!!"""
        # return self._up
        raise Exception("up not implemented")

    @up.setter
    def up(self, value):
        # if len(value) != 3:
        #     sys.stderr.write("ERROR, up must have 3 elements.")
        # self._up = numpy.array(up)
        # self.update_model_matrix()
        raise Exception("up not implemented")

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

    def update_model_matrix(self):
        """(Internal function to update stuff needed by OpenGL.)"""
        q = self._rotation
        s = 1./( (q*q).sum() )
        rot = numpy.matrix(
            [[ 1.-2*s*(q[1]*q[1]+q[2]*q[2]) ,    2*s*(q[0]*q[1]-q[2]*q[3]) ,    2*s*(q[0]*q[2]+q[1]*q[3])],
             [    2*s*(q[0]*q[1]+q[2]*q[3]) , 1.-2*s*(q[0]*q[0]+q[2]*q[2]) ,    2*s*(q[1]*q[2]-q[0]*q[3])],
             [    2*s*(q[0]*q[2]-q[1]*q[3]) ,    2*s*(q[1]*q[2]+q[0]*q[3]) , 1.-2*s*(q[0]*q[0]+q[1]*q[1])]],
            dtype=numpy.float32)
        mat = numpy.matrix( [[ self._scale[0], 0., 0., 0. ],
                             [ 0., self._scale[1], 0., 0. ],
                             [ 0., 0., self._scale[2], 0. ],
                             [ 0., 0., 0., 1.]], dtype=numpy.float32 )
        rotation = numpy.identity(4, dtype=numpy.float32)
        rotation[0:3, 0:3] = rot.T
        mat *= rotation
        translation = numpy.identity(4, dtype=numpy.float32)
        translation[3, 0:3] = self._pos
        mat *= translation
        self.model_matrix[:] = mat
        self.inverse_model_matrix[0:3, 0:3] = numpy.linalg.inv(mat[0:3, 0:3]).T

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
        else:
            self.context.remove_object(self)

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
        # sys.stderr.write("Initializing trail at pos {} with color {}.\n".format(self._pos, self._color))
        self._trail = CylindarStack(color=self._color, maxpoints=self._retain,
                                    points=[ self._pos ], num_edge_points=6)
        # points = numpy.empty( [ self._retain, 3 ] , dtype=numpy.float32 )
        # points[:, :] = self._pos[numpy.newaxis, :]
        # self._trail = Curve(color=self._color, points=points, radius=0.05)
        self._nexttrail = self._interval

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

# ======================================================================

class Box(GrObject):
    """A rectangular solid with dimenions (x,y,z) = (length,height,width)"""
    
    @staticmethod
    def make_box_buffers(context):
        with GrContext._threadlock:
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
#
# The Icosahedron code may seem a little over-verbose.  I originally
# wrote it using indexed buffers, back when I did one GL draw call for
# each object.  (Slow.)  Now that I dump all triangles into one giant
# VAO, I don't use indexed buffers.

class Icosahedron(GrObject):
    """An icosahedron or subdivided icosahedron, with flat or smooth shading (for spheres)."""
    
    @staticmethod
    def make_icosahedron_vertices(subdivisions=0):
        """Internal, do not call."""
        with GrContext._threadlock:
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


    def __init__(self, radius=1., flat=False, subdivisions=0, *args, **kwargs):
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
        super().__init__(subdivisions=subdivisions, *args, **kwargs)


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
        super().__init__(subdivisions=subdivisions, *args, **kwargs)

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
        with GrContext._threadlock:
            if not hasattr(Cylinder, "_vertices"):
                Cylinder._vertices = {}
                Cylinder._normals = {}

        with GrContext._threadlock:
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
        with GrContext._threadlock:
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
            
        # sys.stderr.write("length={:.3f}, shaftl={:.3f}, shaftw={:.3f}, headw={:.3f}, headl={:.3f}\n"
        #                  .format(length, shaftl, shaftw, headw, headl))

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
#
# This is curve is "fixed length" in terms of number of points, not in
# terms of any actual physical dimension.

class FixedLengthCurve(GrObject):
    """A curved tube around a path.  The path is specified by a series of points.

    It's "Fixed Length" because once you create it, you cannot change the number
    of points in the curve.  You may still change the physical dimensions by using
    axis, as with any other object.
    """
    
    def __init__(self, radius=0.05, points=None, *args, **kwargs):
        """Parameters:

        points — A n×3 array that specifies the position of n points.
                 This is the path that is the center of the tube.
        radius — The radius of the tube that will be drawn around the path.

        Plus the other standard GrObject parameters.
        """
        
        super().__init__(*args, **kwargs)

        self._object_type = _OBJ_TYPE_CURVE
        
        self._radius = radius
        if points is None:
            sys.stderr.write("Created an empty curve, doing nothing.\n")
            return

        self._points = numpy.array(points, dtype=numpy.float32)
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise Exception("Illegal points; must be n×3.")
        self._transverse = []
        
        self.make_transverse()

        self.finish_init()

    # Adds a point to the end of the curve
    #   and removes the first point.
    def push_point(self, pos):
        """Add a new point to the end of the curve, and remove the first point from the curve.

        pos — The position of the new point.
        """
        self._points[:-1, :] = self._points[1:, :]
        # error check pos?
        self._points[-1, :] = pos
        self._transverse[:-1, :] = self._transverse[1:, :]
        axishat = self._points[-1, :] - self._points[-2, :]
        axishat /= numpy.sqrt(numpy.square(axishat).sum())
        self._transverse[-1, :] = self._transverse[-2, :] - axishat * (self._transverse[-2, :] * axishat).sum()
        self._transverse[-1, :] /= numpy.sqrt(numpy.square(self._transverse[-1, :]).sum())
        self._transverse[-1, :] *= self._radius
        self.broadcast("update vertices")
        
    @property
    def points(self):
        """The array of points on the curve.  Returned by reference, so be careful."""
        return self._points

    @points.setter
    def points(self, points):
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise Exception("Illegal points; must be n×3.")
        self._points = numpy.array(points, dtype=numpy.float32)
        self.make_transverse()
        self.broadcast("update vertices")

    @property
    def trans(self):
        """The array of transverse vectors that were generated for the curve."""
        return self._transverse
        
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

    def make_transverse(self):
        """Internal, do not call."""
        self._transverse = numpy.empty( self._points.shape , dtype=numpy.float32)
        axes = self._points[1:, :] - self.points[:-1, :]
        axesmag = numpy.sqrt(numpy.square(axes).sum(axis=1))
        # Note: this will div by 0 if any points are doubled
        hatxes = axes / axesmag[:, numpy.newaxis]

        if self._points.shape[0] > 2:
            # All points but first and last

            self._transverse[1:-1, :] = axes[:-1] - axes[1:]
            
            # First and last points : take the adjacent transverse, but then only
            # the component perpendicular to the one axis it's sticking to

            self._transverse[0, : ] = self._transverse[1, :] - hatxes[0] * (self._transverse[1, :] * hatxes[0]).sum()
            self._transverse[-1, :] = self._transverse[-2, :] - hatxes[1] * (self._transverse[-2, :] *
                                                                             hatxes[-1]).sum()
        else:
            # if axis isn't along z, cross z with it get transverse.  Otherwise, cross x with it
            if hatxes[0, 2] < 0.9:
                self._transverse[0:1, :] = numpy.array( [ -hatxes[0, 1], hatxes[0, 0], 0. ], dtype=numpy.float32 )
            else:
                self._transverse[0:1, :] = numpy.array( [ 0., -hatxes[0, 2], hatxes[0, 1] ], dtype=numpy.flat32 )
                
        transmag = numpy.sqrt(numpy.square(self._transverse).sum(axis=1))
        self._transverse *= self._radius / transmag[:, numpy.newaxis]
        
# ======================================================================

class Helix(FixedLengthCurve):
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
        length — The length of the spring (redundant with the length of axis)
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
        # This is what I call "radius" in the FixedLengthCurve class
        self.radius = self._thickness
            
# ======================================================================
# A CylindarStack is not a GrObject, even though some of the interface is the same

class CylindarStack(object):
    """A collection of cylinders whose endpoints aling.  Used to make a trail."""
    
    def __init__(self, radius=0.01, maxpoints=50, color=color, points=None, num_edge_points=6, *args, **kwargs):
        """Parameters:

        radius — The radius of each cylinder
        maxpoints — The number of points in the line.  A cylindar will be drawn between each adjacent pair of points.
                    If you add more points than maxpoints, the first points get removed.
        points — A n×3 vector of points specifying the path covered by cylinders.  Must have n < maxpoints
        num_edge_points — The number of sides in the polygon that makes the end of each cylinder (default: 6)
        color — The color of the stack of cylinders.  (They're all the same color.)

        This object is *not* a GrObject, so the other usual GrObject parameters won't work with it.
        """

        self._pos = vector( [0., 0., 0.] )
        if points is not None:
            self._pos = vector(points[0])
        points = numpy.array(points)
        if len(points) > maxpoints: maxpoints = len(points)
        self.radius = radius
        self.color = color
        self.num_edge_points = num_edge_points

        self.pointbuffer = numpy.empty( [maxpoints, 3], dtype=numpy.float32 )
        self.cylbuffer = [None] * maxpoints

        for i in range(len(points)):
            self.pointbuffer[i, :] = points[i, :]
            if i < len(points)-1:
                axis = points[i+1] - points[i]
                self.cylbuffer[i] = Cylinder(pos=points[i], axis=axis, color=self.color,
                                             radius=radius, num_edge_points=self.num_edge_points, *kargs, **kwargs)

        self.maxpoints = maxpoints
        self.nextpoint = len(points)
        if self.nextpoint > self.maxpoints:
            self.nextpoint = 0

        self._visible = True
        
    def add_point(self, point, *args, **kwargs):
        """Add a point to the path.  If this exceeds the maximum number of points, the first point is removed."""

        self.pointbuffer[self.nextpoint, :] = point[:]
        lastpoint = self.nextpoint - 1
        if lastpoint < 0 : lastpoint = self.maxpoints-1

        axis = self.pointbuffer[self.nextpoint, :] - self.pointbuffer[lastpoint, :]

        if self.cylbuffer[lastpoint] is not None:
            self.cylbuffer[lastpoint].pos = self.pointbuffer[lastpoint]
            self.cylbuffer[lastpoint].axis = axis
        else:
            self.cylbuffer[lastpoint] = Cylinder(pos=self.pointbuffer[lastpoint, :], axis=axis,
                                                 radius=self.radius, num_edge_points=self.num_edge_points,
                                                 color=self.color,
                                                 *args, **kwargs)

        self.nextpoint += 1
        if self.nextpoint >= self.maxpoints: self.nextpoint = 0

    @property
    def pos(self):
        """The position of the object.  This is not necessarily the same as any of the points in the path."""
        return self._pos

    @pos.setter
    def pos(self, value):
        if len(value != 3):
            sys.stderr.write("ERROR, pos must have 3 elements.")
            sys.exit(20)
        offset = numpy.array(pos) - self._pos
        self._pos = vector(pos)
        # This is going to muck about with some uninitialized data, but it doesn't matter.
        self.pointbuffer += offset
        for i in range(self.maxpoints):
            if self.cylbuffer[i] is not None:
                self.cylbuffer[i].pos = self.pointbuffer[i]

    
    @property
    def visible(self):
        """Set to False to remove an object from the display, True to put it back."""
        return self._visible

    @visible.setter
    def visible(self, value):
        if value == self._visible: return

        self._visible = value
        for cyl in self.cylbuffer:
            if cyl is not None:
                cyl.visible = self._visible
            
    def __del__(self):
        self.visible = False

# ======================================================================

def main():
    dobox1 = True
    dobox2 = True
    doball = True
    dopeg = False
    dopeg2 = False
    doblob = False
    doarrow = True
    dohelix = True
    docurve = True
    domanyelongatedboxes = False

    # Make objects
    sys.stderr.write("Making boxes and peg and other things.\n")
    
    if dobox1:
        sys.stderr.write("Making box1.\n")
        box1 = Box(pos=(-0.5, -0.5, 0), length=0.25, width=0.25, height=0.25, color=[0.5, 0., 1.])
    if dobox2:
        sys.stderr.write("Making box2.\n")
        box2 = Box(pos=( 0.5,  0.5, 0), length=0.25, width=0.25, height=0.25, color=color.red)

    if dopeg:
        sys.stderr.write("Making peg.\n")
        peg = Cylinder(pos=(0., 0., 0.), radius=0.125, color=color.orange, num_edge_points=32)
        peg.axis = (0.5, 0.5, 0.5)
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
    

    if docurve:
        sys.stderr.write("Making curve.\n")
        points = numpy.empty( [100, 3] )
        for i in range(100):
            phi = 6*math.pi * i / 50.
            points[i] = [0.375*math.cos(phi), 0.375*math.sin(phi), 1.5 * i*i / 5000. ]
        curve = FixedLengthCurve(radius = 0.05, color = (0.75, 1.0, 0.), points = points)
        
    if dohelix:
        sys.stderr.write("Making helix.\n")
        helix = Helix(color = (0.5, 0.5, 0.), radius=0.2, thickness=0.05, length=2., coils=5,
                      num_circ_points=12)

    if domanyelongatedboxes:
        n = 8
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


    # Updates
    
    theta = math.pi/4.
    phi = 0.
    phi2 = 0.
    fps = 30
    global _print_fps
    _print_fps = True
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
            
            if first:
                box2.interval = 5
                box2.retain = 50
                box2.make_trail = True
                first = False

            # if phi > math.pi:
            #     box2.visible = False
            # else:
            #     box2.visible = True
                
        if doarrow:
            arrow.axis = [math.cos(phi) * (1. + 0.5*math.cos(phi)),
                          math.sin(phi) * (1. + 0.5*math.cos(phi)), 0.]
        
        if dohelix:
            helix.length = 2. + math.cos(phi)

        if docurve:
            curve.radius = 0.05 + 0.04*math.sin(phi)

            if phi2 > math.pi and phi2 < 3.*math.pi/2.:
                curve.visible = False
            else:
                curve.visible = True
                
        if domanyelongatedboxes:
            # Rotate all the elongated boxes
            for i in range(len(boxes)):
                boxes[i].axis = numpy.array( [math.sin(theta)*math.cos(phi+phases[i]),
                                             math.sin(theta)*math.sin(phi+phases[i]),
                                             math.cos(theta)] )

        rate(fps)
        nextprint -= 1
        if nextprint <= 0 :
            nextprint = printfpsevery
            nexttime = time.perf_counter()
            sys.stderr.write("Effective main() fps = {}\n".format(printfpsevery / (nexttime - lasttime)))
            lasttime = nexttime
                             
        



# ======================================================================

if __name__ == "__main__":
    main()

