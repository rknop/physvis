physvis is a Python 3 library for three-dimensional visualization of
simple objects.  You can create balls, cylinders, springs, and move them
around very easily with just a few commands.

Documentation can be found in the "doc" subdirectory; read
physvis.html for the basic information.  (visual_base.html has further
information about the underlying implementation.)  This documentation
was generated from the source code using the utility pydoc (for Python
3).

physvis (c) 2019 by Rob Knop, and is available under the GPL version 3.0
or later; see the file COPYING for the full license.

REQUIREMENTS

physvis works only with Python 3; it will not work with Python 2.

physvis requires numpy (https://www.numpy.org) and PyOpenGL
(http://pyopengl.sourceforge.net/).  If you're on Linux, almost
certainly both of these (for Python 3) are included with your
distribution.

(I have not tried this myself anywhere other than on a couple of Linux
machines, so... good luck.)

USAGE

The easiest way to use this is to copy the two files physvis.py and
visual_base.py to the directory where your own python code is.

You can also put those two files somewhere in your python library path,
or add the directory where they already exist to your python library
path.  How you do that is dependent on your operating system and/or
programming environment you use.

See doc/physvis.html for a very brief introduction to code that uses
this library.  See bouncing_ball.py, rotating_spring.py, and
vibrating_array.py for examples.
