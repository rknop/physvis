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

"""A class designed to support the "rate()" command, which gives the
draw loop time to do stuff and also lets you limit your code to run at a
given fps.  Used when the main loop of the code is the "user code".  If
the context is part of a UI library, them the main thread is (probably)
the UI library's main loop, which should take care of drawing and such.
In that case, this class can still limit the speed at which code runs.
"""

import sys
import time
import threading

class Rater(threading.Event):
    """A singleton class used internally to limit the rate at which a loop runs.

    Get the instance with Rater.get().  Call the rate(fps) method of a Rater
    object to make the code sleep just enough so that it runs only once
    every 1/fps secons.

    """

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
        """Call this to have the program quit the next time you call rate().

        Needs some work.  It's really unelegant to just call sys.exit(),
        but that's what happens.

        """
        
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
            else:
                # Gotta make sure the thread yields!
                # ...doesn't seem to let the drawing happen...
                # time.sleep(0.0001)
                pass
        # This is here because drawing (really, the updating done in the
        #   idle function of the draw thread) will never happen if it
        #   doesn't get some idle time to do so.  This lets the draw
        #   happen even if we're going full tilt and didn't sleep.  The
        #   issue with it is that if the actual rate at which this rate
        #   function is called and the timeout of the idle function is
        #   not well-synchornized, then I think we end up waiting here
        #   for extra time.  I think.  In any event, profiling tells me
        #   that I'm spending time in thread waiting even when the main
        #   calculation goes slower than the rate statement (i.e. when
        #   we're not sleeping).  I have to come up with a better way to
        #   synchoronize things, or at least make sure that the idle
        #   function of the graphics system (GLUT or Qt) main loop
        #   happens.
        self.wait()
        self.clear()
        self._time_of_last_rate_call = time.perf_counter()

