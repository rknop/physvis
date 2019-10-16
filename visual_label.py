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

"""This module handles label objects.  It's separate so that you don't
need to load cairocffi if you don't want to use labels.  """

import sys
import math
import numpy

from grcontext import *
from object_collection import *
from visual_base import *

import cairocffi as cairo


class LabelObject(GrObject):
    """A label is a text string that always faces the camera.

    Some standard object properties are ignored, including rot, scale,
    axis, up.  """

    _known_units = ['display', 'centidisplay', 'pixels']
    _known_fonts = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
    
    _is_little_endian = None

    @staticmethod
    def isLittleE():
        n = 1 + 2*256 + 3*65536 + 5 * 16777216
        bigE = numpy.array( [n], dtype=">i4")
        littleE = numpy.array( [n], dtype="<i4")
        native = numpy.array( [n], dtype="=i4")

        bytebigE = numpy.frombuffer(bigE, dtype=numpy.uint8)
        bytelittleE = numpy.frombuffer(littleE, dtype=numpy.uint8)
        bytenative = numpy.frombuffer(native, dtype=numpy.uint8)

        if (bytelittleE == bytenative).all():
            return True
        elif (bytebigE != bytenative).all():
            raise Exception("Cannot determine if system is big-endian or little-endian.\n")



    def __init__(self, xoffset=0., yoffset=0.,
                 text="test", font="serif", italic=True, bold=False,
                 height=0.25, width=None, units='centidisplay', color=None,
                 border=0.025, box=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with Subject._threadlock:
            if LabelObject._is_little_endian is None:
                LabelObject._is_little_endian = LabelObject.isLittleE()
        
        if not units in LabelObject._known_units:
            raise Exception("Unknown unit \"{}\"".format(unit))
        self._units = units

        if not font in LabelObject._known_fonts:
            raise Exception("Unknown font\"{}\"; known values are {}".format(font, LabelObject._known_fonts))
        self._font = font

        self._height = height
        self._width = width
        self._italic = italic
        self._bold = bold
        self._border = border
        self._box = box
        self._text = text
        self._xoff = xoffset
        self._yoff = yoffset
        self.texturedata = numpy.array( (LabelObjectCollection._TEXTURE_SIZE,
                                         LabelObjectCollection._TEXTURE_SIZE) , dtype=numpy.ubyte )
        
        self.render_text()

        self.finish_init()
        

    @property
    def pos(self):
        """The 3d reference position of the label (a vector).  Set this to move the label."""
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = vector(value)
        self.update_vertices()

    @property
    def x(self):
        """The x-component of reference position."""
        return self._pos[0]

    @x.setter
    def x(self, value):
        self._pos[0] = value
        self.update_vertices()

    @property
    def y(self):
        """The y-component of reference position."""
        return self._pos[1]

    @y.setter
    def y(self, value):
        self._pos[1] = value
        self.update_vertices()

    @property
    def z(self):
        """The z-component of reference position."""
        return self._pos[2]

    @z.setter
    def z(self, value):
        self._pos[2] = value
        self.update_vertices()

    @property
    def xoffset(self):
        return self._xoff

    @xoffset.setter
    def xoffset(self, value):
        self._xoff = value
        self.update_vertices()

    @property
    def yoffset(self):
        return self._yoff

    @yoffset.setter
    def yoffset(self, value):
        self._yoff = value
        self.update_vertices()

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if not value in LabelObject._known_units:
            raise Exception("Unknown unit \"{}\"".format(value))
        self._units = value
        self.render_text()
        self.update_everything()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self._width = None
        self.render_text()
        self.update_everything()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._height = None
        self.render_text()
        self.update_everything()

    @property
    def font(self):
        return self._font

    @font.setter
    def font(self, value):
        if not value in LabelObject._known_fonts:
            raise Exception("Unknown font\"{}\"; known values are {}".format(value, LabelObject._known_fonts))
        if self._font != value:
            self._font = value
            self.render_text()
            self.update_everything()

    @property
    def italic(self):
        return self._italic

    @italic.setter
    def italic(self, value):
        if self._italic != value:
            self._italic = value
            self.render_text()
            self.update_everything()

    @property
    def bold(self):
        return self._bold

    @bold.setter
    def bold(self, value):
        if self._bold != value:
            self._bold = value
            self.render_text()
            self.update_everything()

    @property
    def border(self):
        return self._border

    @border.setter
    def border(self, value):
        if self._border != value:
            self._border = value
            self.render_text()
            self.update_everything()

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, value):
        if self._box != value:
            self._box = value
            self.render_text()
            self.update_everything()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self):
        self._text = text
        self.render_text()
        self.update_everything()


    def update_everything(self):
        self.broadcast("update everything")

    def update_vertices(self):
        self.broadcast("update vertices")

    def render_text(self):
        imwid = LabelObjectCollection._TEXTURE_SIZE
        # Create a memory surface to render to and a drawing context
        img = cairo.IamgeSurface(ciaro.FORMAT_ARGB32, imwid, imwid)
        ctx = cairo.Context(img)

        # Get the font, make it a color
        italic = cairo.FONT_SLANT_ITALIC if self._italic else cairo.FONT_SLANT_NORMAL
        bold = cairo.FONT_WEIGHT_BOLD if self._bold else cairo.FONT_WEIGHT_NORMAL
        font cairo.ToyFontFace(self._font, italic, bold)
        ctx.set_font_face(font)
        ctx.set_source_rgb(self.color)

        # Figure out how big the word is going to be with font size 100
        ctx.set_font_size(100)
        scf = cairo.ScaledFont(font, ctx.get_font_matrix(), ctx.get_matrix(), cairo.FontOptions())
        xbear, ybear, wid, hei, xadv, yadv = scf.text_extents(text)

        # Scale the font so that the width will fill imwid - 6 - 2*border (if there is a border)
        if self._border:
            ctx.set_font_size( (imwid - 6 - 2*self.border) / (wid/100.) )
        else:
            ctx.set_font_size( (imwid - 6) / (wid/100.) )

        # Get a path representing the text, and figure out how big it is
        ctx.text_path(text)
        x0, y0, x1, y1 = ctx.fill_extents()

        # Set the position offset from the center of
        #   the image by half the size of the path,
        #   get a new path at the right place,
        #   figure it what it covers on the image,
        #   and draw it.
        xpos = imwid//2 - (x1-x0)/2
        ypos = imwid//2
        ctx.new_path()
        ctx.move_to(xpos, ypos)
        ctx.text_path(text)
        x0, y0, x1, y1 = ctx.fill_extents()
        ctx.fill()

        if self._box:
            # Draw a box around the text
            # ROB! Think about line width
            ctx.move_to(x0-border, y0-border)
            ctx.line_to(x1+border, y0-border)
            ctx.line_to(x1+border, y1+border)
            ctx.line_to(x0-border, y1+border)
            ctx.close_path()
            ctx.stroke()

        # Make sure the image is fully drawn
        img.flush()

        data = numpy.frombuffer(img.get_data(), dtype=numpy.ubyte)
        data.shape = (imwid, imwid, 4)

        # Move the colors around to what OpenGL needs
        if LabelObject._is_little_endian:
            self.texturedata[:, :, 0] = data[:, :, 2]
            self.texturedata[:, :, 1] = data[:, :, 1]
            self.texturedata[:, :, 2] = data[:, :, 0]
            self.texturedata[:, :, 3] = data[:, :, 3]
        else:
            self.texturedata[:, :, 0:2] = data[:, :, 1:3]
            self.texturedata[:, :, 3] = data[:, :, 0]

        # ROB YOU STOPPED HERE, MORE NEEDS TO BE DONE
        
        
