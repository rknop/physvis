physvis is a Python 3 library for three-dimensional visualization of
simple objects.  You can create balls, cylinders, springs, and move them
around very easily with just a few commands.

Documentation can be found in the "doc" subdirectory; read
physvis.html for the basic information.  (visual_base.html has further
information about the underlying implementation.)  This documentation
was generated from the source code using the utility pydoc (for Python
3).

physvis (c) 2020 by Rob Knop, and is available under the GPL version 3.0
or later; see the file COPYING for the full license.

REQUIREMENTS

physvis works only with Python 3; it will not work with Python 2.

physvis requires numpy (https://www.numpy.org) and PyOpenGL
(http://pyopengl.sourceforge.net/); if you want to use it with the Qt
GUI library, you also need PyQT5
(https://www.riverbankcomputing.com/software/pyqt/intro).  If you're on
Linux, almost certainly both of these (for Python 3) are included with
your distribution.

I've used this on Linux.  It has been succesfully used on Windows with
Anaconda installed.  You can install the needed PyOpenGL libraries with

   conda install pyopengl
   conda install freeglut
   conda install pyqt5

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
  qtgrcontext.py
  quaternions.py
  rater.py
  visual_base.py
  visual_label.py

For cut and paste purposes:

  grcontext.py object_collection.py physvis_observer.py physvis.py qtgrcontext.py quaternions.py rater.py visual_base.py visual_label.py

See doc/physvis.html for a very brief introduction to code that uses
this library.  In the archive there are also various example programs,
in increasing order of complexity:
  bouncing_ball.py
  testfaces.py
  facestest.py  (I'm sure there's a good reason for both this and the previous)
  toomanyspheres.py
  testcurve.py
  axes.py
  many_rotators.py
  rotating_spring.py
  testtwowin.py
  testtwocurves.py (Qt; run with --help to see more info)
  vibrating_array.py (requires scipy; run with argument "qt" to use Qt backend)
