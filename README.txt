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

I've used this on Linux.  It has been succesfully used on Windows with
Anaconda installed.  You can install the needed PyOpenGL libraries with

   conda install pyopengl
   conda install freeglut

USAGE

You need to have all of the following files available to your Python
program.  You can just put them in the same directory with your code,
or, better, you can install them in a "library" directory.  (What those
directories are, and how to add new library directories with environment
variables, will depend on your operating system.)

  grcontext.py
  object_collection.py
  physvis_observer.py
  physvis.py
  quaternions.py
  rater.py
  visual_base.py

See doc/physvis.html for a very brief introduction to code that uses
this library.  See bouncing_ball.py, rotating_spring.py, and
vibrating_array.py for examples.
