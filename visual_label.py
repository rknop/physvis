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
    axis, up.  If you try to set any of these, strange and unexpected
    things will happen.  (I should make it so that doesn't happen.)

    The following properties exist; they may be set directly, or passed
    to the object initialization.

    text — the text to render
    font — one of 'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'
    height — The height of the font -- really, the point size of the font (default: 12)
    italic — True or False to italicize (default: True)
    bold — True or False for bold text (default; False)

    pos — a 3d vector, the reference position for where the label is
    color — three values, the r, g, and b values of the text (between 0 and 1)

    units — units for xoffset, yoffset, refheight.  There are three posibilities.
             "display" means distances correspond to distances in world.
             "centidisplay" (the default) is distances in world * 100
                (so 25 would be 1/4 of one in-world unit).
             "pixels" is not implemented.
    xoffset, yoffset — By default, the label is positioned so that the
           center of the bottom of the label is at the reference point
           specified by pos.  Give these values to offset the label from
           that position; the units are given by the "units" keyword.
    refheight — A "reference height".  The height in units of a
           character in a 12-point font.  Defaults to 25 centidisplay
           units (or 0.25 display units).
    box — Set to True to draw a box around the text (default: True)
    border — Border in points between the text and the box (default: 1)
    linecolor — Color of the border (default: same as text)
    linewidth — Width of the border line in points (default: 1)
    """

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



    def __init__(self, text="test", font="serif", italic=True,
                 bold=False, height=12, units='centidisplay',
                 xoffset=0., yoffset=0., refheight=25,
                 box=True, border=1, linecolor=None, linewidth=1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        with Subject._threadlock:
            if LabelObject._is_little_endian is None:
                LabelObject._is_little_endian = LabelObject.isLittleE()

        self._object_type = GLObjectCollection._OBJ_TYPE_LABEL
        if not units in LabelObject._known_units:
            raise Exception("Unknown unit \"{}\"".format(unit))
        self._units = units

        if not font in LabelObject._known_fonts:
            raise Exception("Unknown font\"{}\"; known values are {}".format(font, LabelObject._known_fonts))
        self._text = text
        self._font = font
        self._italic = italic
        self._bold = bold
        self._height = height
        self._xoff = xoffset
        self._yoff = yoffset
        self._refheight = refheight
        self._box = box
        self._border = border
        self._linecolor = linecolor
        self._linewidth = linewidth

        self.glxoff = 0.
        self.glyoff = 0.
        self.texturedata = numpy.empty( (LabelObjectCollection._TEXTURE_SIZE,
                                         LabelObjectCollection._TEXTURE_SIZE, 4) , dtype=numpy.ubyte )
        
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
    def text(self):
        return self._text

    @text.setter
    def text(self):
        self._text = text
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
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self._width = None
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
    def refheight(self):
        return self._refheight

    @refheight.setter
    def refheight(self, value):
        self._refheight = value
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
    def border(self):
        return self._border

    @border.setter
    def border(self, value):
        if self._border != value:
            self._border = value
            self.render_text()
            self.update_everything()

    def update_everything(self):
        self.broadcast("update everything")

    def update_vertices(self):
        self.update_model_matrix()
        self.broadcast("update vertices")

    def render_text(self):
        imwid = LabelObjectCollection._TEXTURE_SIZE
        # Create a memory surface to render to and a drawing context
        img = cairo.ImageSurface(cairo.FORMAT_ARGB32, imwid, imwid)
        ctx = cairo.Context(img)

        # Get the font, make it a color
        italic = cairo.FONT_SLANT_ITALIC if self._italic else cairo.FONT_SLANT_NORMAL
        bold = cairo.FONT_WEIGHT_BOLD if self._bold else cairo.FONT_WEIGHT_NORMAL
        font = cairo.ToyFontFace(self._font, italic, bold)
        ctx.set_font_face(font)
        # sys.stderr.write("Setting text color to r={}, g={}, b={}\n"
        #                  .format(self.color[0], self.color[1], self.color[2]))
        ctx.set_source_rgb(self.color[0], self.color[1], self.color[2])

        # Figure out how big the word is going to be with font size 100
        ctx.set_font_size(100)
        scf = cairo.ScaledFont(font, ctx.get_font_matrix(), ctx.get_matrix(), cairo.FontOptions())
        xbear, ybear, wid, hei, xadv, yadv = scf.text_extents(self._text)

        # We are given self._height as the point size of the font,self._border in
        # points as the distance from the edge of the word to the
        # border, and self._linewidth in points as the width of the border
        # line.  We want to just fit this into a imwid × imwid image.  Figure out
        # what point size fsize we should tell Cairo to make this work

        if self._box:
            fullwid = wid + 2 * ( (self._border + self._linewidth) * ( 100./self._height ) )
            fullhei = hei + 2 * ( (self._border + self._linewidth) * ( 100./self._height ) )
        else:
            fullwid = wid
            fullhei = hei

        if fullwid > fullhei:
            fsize = imwid / fullwid * 100
        else:
            fsize = imwid / fullhei * 100
            
        ctx.set_font_size(fsize)
        bordersp = self._border * fsize / self._height
        linew = self._linewidth * fsize / self._height

        # Get a path representing the text, and figure out how big it is
        ctx.text_path(self._text)
        x0, y0, x1, y1 = ctx.fill_extents()

        # Set the position offset from the center of
        #   the image by half the size of the path,
        #   get a new path at the right place,
        #   figure it what it covers on the image,
        #   and draw it.
        xpos = imwid//2 - (x1-x0)/2
        ypos = imwid - (y1 + bordersp+linew if self._box else y1 )
        ctx.new_path()
        ctx.move_to(xpos, ypos)
        ctx.text_path(self._text)
        x0, y0, x1, y1 = ctx.fill_extents()
        ctx.fill()

        if self._box:
            if self._linecolor is not None:
                ctx.set_source_rgb(self._linecolor[0], self._linecolor[1], self._linecolor[2])
            udx, udy = ctx.device_to_user_distance(linew, linew)
            if udx > udy:
                ctx.set_line_width(udx)
            else:
                ctx.set_line_width(udy)
            ctx.move_to(x0-bordersp, y0-bordersp)
            ctx.line_to(x1+bordersp, y0-bordersp)
            ctx.line_to(x1+bordersp, y1+bordersp)
            ctx.line_to(x0-bordersp, y1+bordersp)
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


        # Figure out what the width and height of the polygon should be.
        # They get stored in fullwid and fullhei (the renderer will use
        # those two fields)

        factor = 1.
        if self._units == "pixels":
            raise Exception("Pixel units for labels isn't implemented")
        elif self.units == "centidisplay":
            factor = 0.01

        glxoff = factor * self._xoff
        glyoff = factor * self._yoff

        self.fullhei = factor * self._refheight * self._height / 12.
        self.fullwid = self.fullhei

        # ... is there anything else I need to do?
            
        
    def destroy(self):
        raise Exception("OMG ROB!  You need to figure out how to destroy things!")
