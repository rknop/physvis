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

# ======================================================================
#
# One object collection encapsulates a set of objects that can
#  all be drawn with the same shader.

from grcontext import *
from physvis_observer import *

_debug_shaders = False
 
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
      
    Subclasses must implement
       — update_object_vertices(grobject) — update the OpenGL data for grobject's vertices
       — canyoutake(grobject) — returns True or False if the collection can handle the object.  (ROB: Race conditions.)
       — probably other things
       — Must call super().initglstuff() in their __init__

    """

    _OBJ_TYPE_NONE = 0
    _OBJ_TYPE_SIMPLE = 1
    _OBJ_TYPE_CURVE = 2
    _OBJ_TYPE_LABEL = 3

    # OBJ_TYPE_SIMPLE goes with:
    #    SimpleObjectCollection
    #    BasicShader
    #    BasicMaterial
    # OBJ_TYPE_CURVE goes with:
    #    CurveCollection
    #    CurveTubeShader
    #    BasicMaterial
    # OBJ_TYPE_LABEL goes with:
    #    LabelObjectCollection
    #    LabelObjectShader
    #    (No material)
    
    _MAX_OBJS_PER_COLLECTION = 512
    _MAX_GLOBAL_LIGHTS = 8
    
    collection_classes = {}
    
    @staticmethod
    def get_new_collection(obj, context):
        for key in GLObjectCollection.collection_classes:
            if obj._object_type == key:
                return GLObjectCollection.collection_classes[key](context)
        raise Exception("No colletion can handle object of type {}\n".format(obj._object_type))

    @staticmethod
    def register_collection_type(collection_class, objid):
        GLObjectCollection.collection_classes[objid] = collection_class
    
    def __init__(self, context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxnumobjs = GLObjectCollection._MAX_OBJS_PER_COLLECTION
        self.objects = {}
        self.object_index = {}
        self.numobjects = 0

        self.context = context

        self.my_object_type = GLObjectCollection._OBJ_TYPE_NONE
        # sys.stderr.write("Returning from GLObjectCollection.__init__\n")

    def initglstuff(self):
        # sys.stderr.write("In GLObjectColletion.initglstuff; self.shader = {}\n".format(self.shader))
        self.modelmatrixbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
        # 4 bytes per float * 16 floats per object
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 4 * 16 * self.maxnumobjs, None, GL.GL_DYNAMIC_DRAW)

        self.modelnormalmatrixbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
        # 4 bytes per float * 9 floats per object
        #  BUT!  Because of std140 layout, there's actually 12 floats per object,
        #    as the alignment of each row of the matrix is like a vec4 rather than a vec3
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 4 * 12 * self.maxnumobjs, None, GL.GL_DYNAMIC_DRAW)

        self.colorbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.colorbuffer)
        # 4 bytes per float * 4 floats per object
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 4 * 4 * self.maxnumobjs, None, GL.GL_DYNAMIC_DRAW)

        self.speculardatabuf = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.sepculardatabuf)
        # (4 byte float + 4 byte int) per object
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 8*self.maxnumobjs, None, GL.GL_DYNAMIC_DRAW)

        self.globallightbuf = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.globallightbuf)
        # (2 * (16-byte vec3 (std140 = vec4 alignment) ) per object
        GL.glBufferData(GL.GL_UNIFORM_BUFFER, 32*self.GLObjectCollection._MAX_GLOBAL_LIGHTS, None, GL.GL_DYNAMIC_DRAW)
        
        try:
            dex = GL.glGetUniformBlockIndex(self.shader.progid, "ModelMatrix")
            GL.glUniformBlockBinding(self.shader.progid, dex, 0);
        except:
            sys.stderr.write("self.shader.progid = {}\n".format(self.shader.progid))
            import pdb; pdb.set_trace()

        dex = GL.glGetUniformBlockIndex(self.shader.progid, "ModelNormalMatrix")
        GL.glUniformBlockBinding(self.shader.progid, dex, 1)

        dex = GL.glGetUniformBlockIndex(self.shader.progid, "Colors")
        GL.glUniformBlockBinding(self.shader.progid, dex, 2)

        dex = GL.glGetUniformBlockIndex(self.shader.progid, "SpecularData")
        GL.glUniformBlockBinding(self.shader.progid, dex, 3)
        
        dex = GL.glGetUniformBlockIndex(self.shader.progid, "GlobalLights")
        GL.glUniformBlockBinding(self.shader.progid, dex, 4)
        
        self.bind_uniform_buffers()
        
        # In the past, I was passing a model matrix for each
        # and every vertex.  That was profligate.  I'm leaving this
        # comment here, though, as it's got a pointer to docs how I made that work.
        # See https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_instanced_arrays.txt
        # and http://sol.gfxile.net/instancing.html

    def canyoutake(self, obj):
        return False
        
    def remove_object(self, obj):
        if not obj._id in self.objects:
            return

        self.context.run_glcode(lambda : self.do_remove_object(obj))

    def bind_uniform_buffers(self):
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, self.modelmatrixbuffer)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 1, self.modelnormalmatrixbuffer)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 2, self.colorbuffer)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 3, self.speculardatabuf)
        GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 4, self.globallightbuf)
        
    def update_object_matrix(self, obj):
        if not obj.visible: return

        if not obj._id in self.objects:
            sys.stderr.write("...object not found whose matrix was to be updated!!\n")
            return

        self.context.run_glcode(lambda : self.do_update_object_matrix(obj))

    def do_update_object_matrix(self, obj):
        with Subject._threadlock:
            if not obj._id in self.objects:
                return
            # sys.stderr.write("Pushing object matrix:\n{}\n".format(obj.model_matrix))
            dex = self.object_index[obj._id]
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelmatrixbuffer)
            GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*16, obj.model_matrix.flatten())
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.modelnormalmatrixbuffer)
            GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, dex*4*12, obj.inverse_model_matrix.flatten())
            self.context.update()

    def do_remove_object_uniform_buffer_data(self, obj):
        with Subject._threadlock:
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
        with Subject._threadlock:
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
        if message == "update material":
            raise Exception("ROB!  Implement update material.")
        if message == "update matrix":
            self.update_object_matrix(subject)
        if message == "update vertices":
            self.update_object_vertices(subject)
        if message == "update everything":
            self.context.run_glcode(lambda : self.push_all_object_info(subject))

# ======================================================================
# SimpleObjectCollection
#
# This is for objects that don't require a geometry shader (so no curves).
# Objects are collections of triangles, and all have a single color.
#
# Shaders take as input for each vertex of each triangle
#  location  (4 floats per vertex; 4th element should be 1)
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
#  color     (4 floats per vertex)  (4th is opacity, but is ignored)
#
#  * really, it's a mat3, so you'd think 9 floats per object.  However,
#  The OpenGL std140 layout means that things are aligned on vec4
#  bounadries, so there's an extra "junk" float at the end of each
#  row of the matrix.

class SimpleObjectCollection(GLObjectCollection):
    """A collection of "simple" objects.

    Each object is represented by a number of triangles.  An object that
    goes into one of these collections must have:
      — num_triangles : the number of triangles in the object
      — vertexdata : a numpy array of float32 that has a sequence of vec4 (w=1), 3 for each triangle (w = 1)
      — normaldata : a numpy array of float32 that has a sequence of vec3, 3 for each triangle

    They must also meet the requirements of GLObjectCollection.
    """
    
    def __init__(self, context, *args, **kwargs):
        # sys.stderr.write("Creating SimpleObjectCollection.\n")
        super().__init__(context, *args, **kwargs)
        self.shader = Shader.get("Basic Shader", context)

        self.maxnumtris = 65536

        self.curnumtris = 0
        self.object_triangle_index = {}

        self.pending_objects = 0
        self.pending_tris = 0
        
        self.draw_as_lines = False

        self.is_initialized = False

        self.my_object_type = GLObjectCollection._OBJ_TYPE_SIMPLE

        # sys.stderr.write("About to call super().initglstuff; self.shader = {}\n".format(self.shader))
        # sys.stderr.flush()
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

        self.is_initialized = True
        self.bind_vertex_attribs()

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

    def canyoutake(self, obj):
        if obj._object_type != self.my_object_type:
            return False
        if len(self.objects) >= self.maxnumobjs:
            return False
        if self.curnumtris + self.pending_tris + obj.num_triangles > self.maxnumtris:
            return False
        return True
        
    def add_object(self, obj):
        # Make sure not to double-add
        if obj._id in self.objects:
            return

        if not self.canyoutake(obj):
            raise Exception("Error, can't add object, limits reached.")

        with Subject._threadlock:
            self.pending_objects += 1
            self.pending_tris += obj.num_triangles

            self.context.run_glcode(lambda : self.do_add_object(obj))
            

    def do_add_object(self, obj):
        with Subject._threadlock:
            if obj._id in self.objects:
                return
            self.object_triangle_index[obj._id] = self.curnumtris
            self.objects[obj._id] = obj
            self.curnumtris += obj.num_triangles
            self.object_index[obj._id] = len(self.objects) - 1
            self.push_all_object_info(obj)
            obj.add_listener(self)

            self.pending_objects -= 1
            self.pending_tris -= obj.num_triangles
            # sys.stderr.write("Up to {} objects, {} triangles.\n".format(len(self.objects), self.curnumtris))

    def do_remove_object(self, obj):
        with Subject._threadlock:
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
                
    # Updates positions of verticies and directions of normals.  Can NOT
    # change the number of vertices.
    def update_object_vertices(self, obj):
        if not obj.visible: return
        if not obj._id in self.objects: return
        self.context.run_glcode(lambda : self.do_update_object_vertex(obj))

    def do_update_object_vertex(self, obj):
        with Subject._threadlock:
            if not obj._id in self.objects:
                return
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertexbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*4*3, obj.vertexdata)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normalbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*3*3, obj.normaldata)
            self.context.update()


    def push_all_object_info(self, obj):
        with Subject._threadlock:
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
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*4*3, obj.vertexdata.flatten())

            # sys.stderr.write("Pushing normaldata for obj {}\n".format(dex))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.normalbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, self.object_triangle_index[obj._id]*4*3*3, obj.normaldata.flatten())

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
        with Subject._threadlock:
            # sys.stderr.write("Drawing Simple Object Collection with shader progid {}\n".format(self.shader.progid))
            GL.glUseProgram(self.shader.progid)
            self.bind_uniform_buffers()
            self.bind_vertex_attribs()
            self.shader.set_camera_perspective()
            self.shader.set_camera_posrot()

            if self.draw_as_lines:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            else:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glBindVertexArray(self.VAO)
            # sys.stderr.write("About to draw {} triangles\n".format(self.curnumtris))
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.curnumtris*3)
            # sys.stderr.write("...done drawing triangles.")


GLObjectCollection.register_collection_type(SimpleObjectCollection, GLObjectCollection._OBJ_TYPE_SIMPLE)

# ======================================================================
# LabelObjectCollection
#
# A label is drawn as two triangles that make a square.  The label has
# an "object position" which is the reference point in 3d space; it
# refers to the middle-bottom of the text (or the surrounding box if
# there is one).  It also includes offsets and sizes which are in view
# space (not clip space).

class LabelObjectCollection(GLObjectCollection):
    """A collection of labels that face the camera at all times."""

    _MAX_LABELS_PER_COLLECTION = 64
    _TEXTURE_SIZE = 256
    
    def __init__(self, context, *args, **kwargs):
        super().__init__(context, *args, **kwargs)
        self.shader = Shader.get("Label Object Shader", context)

        self.label_texture_index = {}
        self.texture_spot_used = numpy.empty( (LabelObjectCollection._MAX_LABELS_PER_COLLECTION), dtype=bool)
        self.texture_spot_used[:] = False

        self.pending_labels = 0

        self.is_initialized = False
        self.my_object_type = GLObjectCollection._OBJ_TYPE_LABEL

        super().initglstuff()

        self.texturearray = GL.glGenTextures(1)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.texturearray)
        GL.glTexStorage3D(GL.GL_TEXTURE_2D_ARRAY, 1, GL.GL_RGBA8,
                          LabelObjectCollection._TEXTURE_SIZE, LabelObjectCollection._TEXTURE_SIZE,
                          LabelObjectCollection._MAX_LABELS_PER_COLLECTION)

        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        
        self.labelposbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.labelposbuffer)
        # 4 bytes per float * 2 floats per vertex * 3 vertices per triange * 2 triangles per label
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4*2*3*2 * LabelObjectCollection._MAX_LABELS_PER_COLLECTION,
                        None, GL.GL_STATIC_DRAW)
        
        self.texcoordbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texcoordbuffer)
        # 4 bytes per float * 2 floats per vertex * 3 vertices per triangle * 2 triangles per label
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4*2*3*2 * LabelObjectCollection._MAX_LABELS_PER_COLLECTION,
                        None, GL.GL_STATIC_DRAW)

        self.objindexbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        # 4 bytes per int * 1 int per vertex * 3 vertices per triangle * 2 triangles per label
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4*1*3*2 * LabelObjectCollection._MAX_LABELS_PER_COLLECTION,
                        None, GL.GL_STATIC_DRAW)

        self.texindexbuffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texindexbuffer)
        # 4 bytes per int * 1 int per vertex * 3 vertices per triangle * 2 triangles per label
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 4*1*3*2 * LabelObjectCollection._MAX_LABELS_PER_COLLECTION,
                        None, GL.GL_STATIC_DRAW)

        self.VAO = GL.glGenVertexArrays(1)
        self.bind_vertex_attribs()
        self.is_initialized = True

    def bind_vertex_attribs(self):
        GL.glBindVertexArray(self.VAO)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.labelposbuffer)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texcoordbuffer)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
        GL.glVertexAttribIPointer(2, 1, GL.GL_INT, 0, None)
        GL.glEnableVertexAttribArray(2)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texindexbuffer)
        GL.glVertexAttribIPointer(3, 1, GL.GL_INT, 0, None)
        GL.glEnableVertexAttribArray(3)


    def canyoutake(self, obj):
        if obj._object_type != self.my_object_type:
            return False
        if len(self.objects) >= LabelObjectCollection._MAX_LABELS_PER_COLLECTION + self.pending_labels:
            return False
        return True

    def add_object(self, obj):
        # Make sure not to double-add
        # sys.stderr.write("Adding a label...\n")
        if obj._id in self.objects:
            return

        if not self.canyoutake(obj):
            raise Exception("Error, can't add label, limits reached.")

        with Subject._threadlock:
            self.pending_labels += 1
            self.context.run_glcode(lambda : self.do_add_object(obj))

    def do_add_object(self, obj):
        with Subject._threadlock:
            if obj._id in self.objects:
                return
            self.objects[obj._id] = obj
            self.object_index[obj._id] = len(self.objects) - 1
            for i in range(self.texture_spot_used.shape[0]):
                if not self.texture_spot_used[i]:
                    self.label_texture_index[obj._id] = i
                    self.texture_spot_used[i] = True
                    break
            self.push_all_object_info(obj)
            obj.add_listener(self)

            self.pending_labels -= 1

    def do_remove_object(self, obj):
        if not obj._id in self.objects:
            return
        dex = self.object_index[obj._id]
        if dex < len(self.objects)-1:
            numafter = len(self.objects) - dex
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.labelposbuffer)
            data = GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, (dex+1)*6*2*4, numafter*6*2*4 )
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*2*4, data)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texcoordbuffer)
            data = GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, (dex+1)*6*2*4, numafter*6*2*4 )
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*2*4, data)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
            data = numpy.empty( numafter * 6, dtype=numpy.int32 )
            GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, (dex+1)*6*1*4, numafter*6*1*4,
                                   ctypes.c_void_p(data.__array_interface__['data'][0]) )
            data[:] -= 1
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*1*4, data)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texindexbuffer)
            GL.glGetBufferSubData( GL.GL_ARRAY_BUFFER, (dex+1)*6*1*4, numafter*6*1*4,
                                   ctypes.c_void_p(data.__array_interface__['data'][0]) )
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*1*4, data)

            for objid in self.objects:
                if self.object_index[objid] > dex:
                    self.object_index[objid] -= 1

            self.do_remove_object_uniform_buffer_data(obj)
                    
        self.texture_spot_used[self.label_texture_index[obj._id]] = False

        del self.objects[obj._id]
        del self.label_texture_index[obj._id]
        obj.remove_listener(self)

        self.context.update()
            

    def update_object_vertices(self, obj):
        if not obj.visible: return
        if not obj._id in self.objects: return
        self.context.run_glcode(lambda : self.push_object_vertices(obj) )

    def push_object_vertices(self, obj):
        with Subject._threadlock:
            if not obj._id in self.objects:
                return
            dex = self.object_index[obj._id]

            # triangle positions
            # I'm worried about left hand forward
            tris = numpy.ones( (6, 2) , dtype=numpy.float32)
            tris[0, 0] = obj.glxoff - obj.fullwid/2.
            tris[0, 1] = obj.glyoff + obj.fullhei
            tris[1, 0] = obj.glxoff + obj.fullwid/2.
            tris[1, 1] = obj.glyoff + obj.fullhei
            tris[2, 0] = obj.glxoff - obj.fullwid/2.
            tris[2, 1] = obj.glyoff
            tris[3, 0] = obj.glxoff + obj.fullwid/2.
            tris[3, 1] = obj.glyoff + obj.fullhei
            tris[4, 0] = obj.glxoff + obj.fullwid/2.
            tris[4, 1] = obj.glyoff
            tris[5, 0] = obj.glxoff - obj.fullwid/2.
            tris[5, 1] = obj.glyoff
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.labelposbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*2*4, tris)
            
            # sys.stderr.write("Pushing object vertices for text at ({}, {}, {}) with "
            #                  "offset ({}, {}) and size ({}, {})\n"
            #                  .format(obj.pos[0], obj.pos[1], obj.pos[2],
            #                          obj.glxoff, obj.glyoff,
            #                          obj.fullwid, obj.fullhei) )
            # for i in range(6):
            #     sys.stderr.write(" ({:6.2f}, {:6.2f})\n".format(tris[i, 0], tris[i, 1]))

            # texture coordinates
            #
            # I don't understsand this: when I swap 1<->0 in y, the text renders
            # exactly the same way.  I would think it would flip the thing vertically
            texcoords = numpy.array( [0., 0.,
                                      1., 0.,
                                      0., 1.,
                                      1., 0.,
                                      1., 1.,
                                      0., 1.] , dtype=numpy.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texcoordbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*2*4, texcoords)

    def push_all_object_info(self, obj):
        with Subject._threadlock:
            sys.stderr.flush()
            if not obj._id in self.objects:
                sys.stderr.write("LabelObjectCollection: tried to push object info for label not in collection")
                sys.stderr.flush()
                return
            dex = self.object_index[obj._id]
            texdex = self.label_texture_index[obj._id]

            sys.stderr.flush()
            self.push_object_vertices(obj)
            sys.stderr.flush()

            # object index
            objindexcopies = numpy.empty(6, dtype=numpy.int32)
            objindexcopies[:] = dex
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.objindexbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*4, objindexcopies)

            # texture index
            texindexcopies = numpy.empty(6, dtype=numpy.int32)
            texindexcopies[:] = texdex
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texindexbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*4, texindexcopies)
            
            # texture coordinates
            texcoords = numpy.array( [0., 0.,
                                      1., 0.,
                                      0., 1.,
                                      1., 0.,
                                      1., 1.,
                                      0., 1.] , dtype=numpy.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.texcoordbuffer)
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, dex*6*2*4, texcoords)

            # texture image data
            # ####
            # sys.stderr.write("Writing texture image to test.ppm, alpha in test.pgm\n")
            # ofppm = open("test.ppm", "wb")
            # ofpgm = open("test.pgm", "wb")
            # ofppm.write("P6 {sz} {sz} 255\n".format(sz=LabelObjectCollection._TEXTURE_SIZE).encode("ascii"))
            # ofpgm.write("P5 {sz} {sz} 255\n".format(sz=LabelObjectCollection._TEXTURE_SIZE).encode("ascii"))
            # for j in range(LabelObjectCollection._TEXTURE_SIZE):
            #     for i in range(LabelObjectCollection._TEXTURE_SIZE):
            #         ofppm.write(obj.texturedata[j, i, (0, 1, 2)])
            #         ofpgm.write(obj.texturedata[j, i, 3])
            # ofppm.close()
            # ofpgm.close()
            # ####
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, self.texturearray)
            GL.glTexSubImage3D(GL.GL_TEXTURE_2D_ARRAY, 0, 0, 0, texdex,
                               LabelObjectCollection._TEXTURE_SIZE, LabelObjectCollection._TEXTURE_SIZE, 1,
                               GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, obj.texturedata)

            self.do_update_object_matrix(obj)
            # Color is irrelevant here

            self.context.update()   # Redundant... just happened in do_update_object_matrix

    # Never call this directly!  It should only be called from within the
    #   draw method of a GrContext
    def draw(self):
        with Subject._threadlock:
            GL.glUseProgram(self.shader.progid)
            GL.glEnable(GL.GL_BLEND);
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);  
            self.bind_uniform_buffers()
            self.bind_vertex_attribs()
            self.shader.set_camera_perspective()
            self.shader.set_camera_posrot()

            GL.glBindVertexArray(self.VAO)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.objects)*6)
        
            
GLObjectCollection.register_collection_type(LabelObjectCollection, GLObjectCollection._OBJ_TYPE_LABEL)
        
# ======================================================================
# CurveCollection

class CurveCollection(GLObjectCollection):
    """A collection of curves defined by a sequence of points.

    Pass the name of the shader in "shader"; defaults to "Curve Tube Shader".
    (Right now, that's the only option.)

    ROB WRITE MORE
    """
    
    def __init__(self, context, shader="Curve Tube Shader", *args, **kwargs):
        # sys.stderr.write("Creating CurveCollection\n")
        super().__init__(context, *args, **kwargs)
        self.shader = Shader.get(shader, context)
        self.maxnumlines=16384

        self.curnumlines = 0
        self.line_index = {}

        self.pending_objects = 0
        self.pending_lines = 0

        self.draw_as_lines = False
        
        self.is_initialized = False

        self.my_object_type = GLObjectCollection._OBJ_TYPE_CURVE
        
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

    def canyoutake(self, obj):
        if obj._object_type != self.my_object_type:
            return False
        if len(self.objects) >= self.maxnumobjs:
            return False
        if self.curnumlines + self.pending_lines + obj.points.shape[0]-1 > self.maxnumlines:
            return False
        return True
        
    def add_object(self, obj):
        if obj._id in self.objects:
            return

        if not self.canyoutake(obj):
            raise Exception("Error, can't add curve, wrong type or limits reached.")

        with Subject._threadlock:
            self.pending_objects += 1
            self.pending_lines += obj.points.shape[0]-1
            self.context.run_glcode(lambda : self.do_add_object(obj))

    def do_add_object(self, obj):
        with Subject._threadlock:
            self.objects[obj._id] = obj
            self.line_index[obj._id] = self.curnumlines
            obj.add_listener(self)
            self.curnumlines += obj.points.shape[0]-1
            # sys.stderr.write("Up to {} curves, {} curve segments.\n".format(len(self.objects), self.curnumlines))

            n = len(self.objects) - 1
            self.object_index[obj._id] = n
            self.push_all_object_info(obj)

            self.pending_objects -= 1
            self.pending_lines -= obj.points.shape[0]-1
        
    def do_remove_object(self, obj):
        with Subject._threadlock:
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

    # Don't change the number of points in the line from when you first
    #   added the object, or things will go haywire.
    def do_update_object_points(self, obj):
        with Subject._threadlock:
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
        with Subject._threadlock:
            # sys.stderr.write("Drawing Curve Tube Collection with shader progid {}\n".format(self.shader.progid))
            GL.glUseProgram(self.shader.progid)
            self.bind_uniform_buffers()
            self.bind_vertex_attribs()
            self.shader.set_camera_perspective()
            self.shader.set_camera_posrot()
            if self.draw_as_lines:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            else:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glBindVertexArray(self.VAO)
            # sys.stderr.write("About to draw {} lines\n".format(self.curnumlines))
            GL.glDrawArrays(GL.GL_LINES, 0, self.curnumlines*2)
            # sys.stderr.write("...done drawing lines\n");

GLObjectCollection.register_collection_type(CurveCollection, GLObjectCollection._OBJ_TYPE_CURVE)
        
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
    _label_shader = {}

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
            with Subject._threadlock:
                if ( (not context in Shader._basic_shader) or
                     (Shader._basic_shader[context] == None) ):
                    Shader._basic_shader[context] = BasicShader(context)
            return Shader._basic_shader[context]

        elif name == "Curve Tube Shader":
            with Subject._threadlock:
                if ( (not context in Shader._curvetube_shader) or
                     (Shader._curvetube_shader[context] == None) ):
                    Shader._curvetube_shader[context] = CurveTubeShader(context)
            return Shader._curvetube_shader[context]

        elif name == "Label Object Shader":
            with Subject._threadlock:
                if ( (not context in Shader._label_shader) or
                     (Shader._label_shader[context] == None) ):
                     Shader._label_shader[context] = LabelObjectShader(context)
            return Shader._label_shader[context]
        
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

        self.ambientcolor = (0.2, 0.2, 0.2)
        self.nlights = 2
        self.lightcolor = [ (0.8, 0.8, 0.8), (0.3, 0.3, 0.3) ]
        self.lightdir = [ (0.22, 0.44, 0.88), (-0.88, -0.22, -0.44) ]
        
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



    def init_lights_and_camera(self):
        # sys.stderr.write("Shader: init_lights_and_camera\n")
        self.update_lights()
        self.set_camera_perspective()
        self.set_camera_posrot()

    def update_lights(self):
        loc = GL.glGetUniformLocation(self.progid, "ambientcolor")
        GL.glUniform3fv(loc, 1, numpy.array(self.ambientcolor, dtype=numpy.float32))

        lightdata = numpy.empty( 8*self.nlights, dtype=numpy.float32 )
        for i in range(self.nlights):
            lightdata[8*i   : 8*i+3] = self.lightcolor[i]
            lightdata[8*i+4 : 8*i+7] = self.lightdir[i]
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self.globallightbuf)
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, 0, 32*self.nlights, lightdata)

    def set_camera_perspective(self):
        GL.glUseProgram(self.progid)
        projection_location = GL.glGetUniformLocation(self.progid, "projection")
        GL.glUniformMatrix4fv(projection_location, 1, GL.GL_FALSE, self.context._perpmat)
        self.context.update()
        
    def set_camera_posrot(self):
        GL.glUseProgram(self.progid)
        viewrot_location = GL.glGetUniformLocation(self.progid, "viewrot")
        GL.glUniformMatrix4fv(viewrot_location, 1, GL.GL_FALSE, self.context._camrotate.T)
        viewshift_location = GL.glGetUniformLocation(self.progid, "viewshift")
        GL.glUniformMatrix4fv(viewshift_location, 1, GL.GL_FALSE, self.context._camtranslate.T)

        self.context.update()

# ======================================================================
 # This shader goes with OBJ_TYPE_SIMPLE and SimpleObjectCollection

class BasicShader(Shader):
    """Shader class for SimpleObjectCollection.  (Render lots of triangles.)"""
    
    def __init__(self, context, *args, **kwargs):
        super().__init__(context, *args, **kwargs)
        self._name = "Basic Shader"

        err = GL.glGetError()

        vertex_shader = """
#version 330

uniform mat4 viewshift;
uniform mat4 viewrot;
uniform mat4 projection;

layout (std140) uniform ModelMatrix
{{
   mat4 model_matrix[{maxnumobj}];
}};

layout (std140) uniform ModelNormalMatrix
{{
   mat3 model_normal_matrix[{maxnumobj}];
}};

layout (std140) uniform Colors
{{
   vec4 color[{maxnumobj}];
}};

layout (std140) uniform SpecularData
{{
   float specstr[{maxnumobj}];
   int specpow[{maxnumobj}];
}};


layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Normal;
layout(location=2) in int in_Index;
out vec3 fragPos;
out vec3 aNormal;
out vec4 aColor;
out float specularStrength;
out int specularPower;

void main(void)
{{
  vec4 worldpos = model_matrix[in_Index] * in_Position 
  fragPos = vec3(worldpos);
  gl_Position =  projection * viewrot * viewshift * worldpos;
  aNormal = model_normal_matrix[in_Index] * in_Normal;
  aColor = color[in_Index];
  specularStrength = specstr[in_Index];
  specularPower = specpow[in_Index];
}}""".format(maxnumobj=GLObjectCollection._MAX_OBJS_PER_COLLECTION)

        fragment_shader = """
#version 330

uniform vec3 ambientcolor;

uniform int numgloblights;
layout (std140) uniform GlobalLights
{{
  vec3 globlightcolor[{maxnumlights}];
  vec3 globlightdir[{maxnumlights}];
}}

in vec3 fragPos;
in vec3 aNormal;
in vec4 aColor;
in float specularStrength;
in int specularPower;
out vec4 out_Color;

void main(void)
{
  vec3 norm = normalize(aNormal);
  vec3 col = ambientcolor
  vec3 viewdir = normalize(-fragPos);
  for (int i = 0 ; i < numgloblights ; ++i)
  {
    vec3 diff = max(dot(norm, globlightdir[i]), 0.) * globlightcolor[i];
    vec3 reflectdir = reflect(-globlightdir[i], norm);
    vec3 spec = specularStrength * pow(max(dot(viewdir, reflectdir), 0.), specularPower) * globlightcolor[i];
    col += spec + diff;
  }
  col *= vec(aColor);
  out_Color = vec4(col, aColor[3]);
}""".format(maxnumlights = GLObjectCollection._MAX_GLOBAL_LIGHTS)

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
# This goes with _OBJ_TYPE_LABEL and LabelObjectCollection

class LabelObjectShader(Shader):
    """Shader class for labels.  Renders a billboard facing the camera."""

    def __init__(self, context, *args, **kwargs):
        super().__init__(context, *args, **kwargs)
        self._name = "Label Object Shader"

        err = GL.glGetError()

        vertex_shader = """
#version 330

uniform mat4 viewshift;
uniform mat4 viewrot;
uniform mat4 projection;

layout (std140) uniform ModelMatrix
{{
   mat4 model_matrix[{maxnumobj}];
}};

layout (std140) uniform ModelNormalMatrix
{{
   mat3 model_normal_matrix[{maxnumobj}];
}};

layout (std140) uniform Colors
{{
   vec4 color[{maxnumobj}];
}};

layout(location=0) in vec2 label_Position;
layout(location=1) in vec2 texCoord;
layout(location=2) in int in_Index;
layout(location=3) in int in_tex_Index;
out vec2 uv;
flat out int texIndex;

void main(void)
{{
  vec4 objpos = viewrot * viewshift * model_matrix[in_Index] * vec4(0., 0., 0., 1.);
  gl_Position = projection * vec4( objpos.x + label_Position.x*(-objpos.z) , 
                                   objpos.y + label_Position.y*(-objpos.z) , objpos.z, 1. );
  // gl_Position = projection * viewrot * viewshift * model_matrix[in_Index] * 
  //               ( obj_Position + vec4(label_Position.x, label_Position.y, 0., 0. ) );
  texIndex = in_tex_Index;
  uv = texCoord;
}}""".format(maxnumobj=GLObjectCollection._MAX_OBJS_PER_COLLECTION)

        fragment_shader = """
#version 330

uniform vec3 ambientcolor;
uniform vec3 light1color;
uniform vec3 light1dir;
uniform vec3 light2color;
uniform vec3 light2dir;

uniform sampler2DArray label_textures;

in vec2 uv;
flat in int texIndex;
out vec4 out_Color;

void main(void)
{
  vec4 texcolor = texture(label_textures, vec3(uv.x, uv.y, texIndex));
  if (texcolor.a < 0.2)
     discard;
  out_Color = texcolor;
}
"""

        # I've cut and paste the rest of this from BasicShader...
        #   this begs some refactoring

        if _debug_shaders: sys.stderr.write("\nAbout to compile label vertex shader....\n")
        self.vtxshdrid = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(self.vtxshdrid, vertex_shader)
        GL.glCompileShader(self.vtxshdrid)

        if _debug_shaders: sys.stderr.write("{}\n".format(GL.glGetShaderInfoLog(self.vtxshdrid)))

        if _debug_shaders: sys.stderr.write("\nAbout to compile label fragment shader....\n")
        self.fragshdrid = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(self.fragshdrid, fragment_shader)
        GL.glCompileShader(self.fragshdrid)

        if _debug_shaders: sys.stderr.write("{}\n".format(GL.glGetShaderInfoLog(self.fragshdrid)))

        if _debug_shaders: sys.stderr.write("\nAbout to link shaders....\n")
        self.progid = GL.glCreateProgram()
        GL.glAttachShader(self.progid, self.vtxshdrid)
        GL.glAttachShader(self.progid, self.fragshdrid)
        GL.glLinkProgram(self.progid)

        if GL.glGetProgramiv(self.progid, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            sys.stderr.write("{}\n".format(GL.glGetProgramInfoLog(self.progid)))
            sys.exit(-1)

        GL.glUseProgram(self.progid)

        if _debug_shaders: sys.stderr.write("Label Object Shader created with progid {}\n".format(self.progid))
        
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

        err = GL.glGetError()
      
        vertex_shader = """
#version 330

uniform mat4 viewshift;
uniform mat4 viewrot;
uniform mat4 projection;

layout (std140) uniform ModelMatrix
{{
   mat4 model_matrix[{maxnumobj}];
}};

layout (std140) uniform ModelNormalMatrix
{{
   mat3 model_normal_matrix[{maxnumobj}];
}};

layout (std140) uniform Colors
{{
   vec4 color[{maxnumobj}];
}};

layout(location=0) in vec4 in_Position;
layout(location=1) in vec3 in_Transverse;
layout(location=2) in int in_Index;
out vec3 aTransverse;
out vec4 aColor;

void main(void)
{{
  gl_Position =  model_matrix[in_Index] * in_Position;
  // aTransverse = model_normal_matrix[in_Index] * in_Transverse;
  vec4 tmp = vec4(in_Transverse, 0);
  tmp = model_matrix[in_Index] * tmp;
  aTransverse = tmp.xyz;
  aColor = color[in_Index];
}}""".format(maxnumobj=GLObjectCollection._MAX_OBJS_PER_COLLECTION)

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

class BasicMaterial(Subject):

    default_specular_strength = 0.5
    default_specular_exponent = 32
    default_color = (1., 1., 1.)

    def __init__(self, specstr=None, spectight=None, color=None):
        if specstr is None:
            self._specstr = BasicMaterial.default_specular_strength
        else:
            self.specstr = specstr
        if spectight is None:
            self._spectight = BasicMaterial.default_specular_exponent
        else:
            self.spectight = spectight
        if color is None:
            self._color = BasicMaterial.default_color
        else:
            self.color = color

    @property
    def specstr(self):
        return self._specstr

    @specstr.setter
    def specstr(self, value):
        if float(value) != self._specstr:
            self._specstr = float(value)
            self.broadcast("update material")

    @property
    def specular_strength(self):
        return self._specstr

    @specular_strength.setter
    def specular_strength(self, value):
        self.specstr = value

    @propery
    def spectight(self):
        return self._spectight

    @spectight.setter
    def spectight(self, value):
        val = int(value)
        if val < 0:
            val = 0
        if val > 255:
            val = 255
        if self._spectight != val:
            self._spectight = val
            self.broadcast("update material")

    @property
    def specular_tightness(self):
        return self._spectight

    @specular_tightness.setter
    def specular_tightness(self, value):
        self.spectight = value
        

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if len(value) != 1 and len(value) != 3:
            raise Exception("Color requires 1 or 3 values.")
        if len(value) == 1:
            newcolor = (float(value), float(value), float(value))
        else:
            newcolor = (float(value[0]), float(value[1]), float(value[2]))

        if self._color != newcolor:
            self._color = newcolor
            self.broadcast("update material")
