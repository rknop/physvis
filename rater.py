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
                # Let other threads have a go
                time.sleep(0)
                pass

        # This .wait() here is to make sure that the drawing thread can
        #   go.  The sleep(0) above isn't enough (or, even, it turns
        #   out, the other sleep, if it's short enough) because the
        #   frequency of drawing might be synced to a monitor's vertical
        #   blank frequency.  That means that if rater isn't sleeping
        #   enough here, the very brief time it yields to other threads
        #   might not include the instant that the drawing thread wants
        #   to go.  As such, we have to wait here so that we know we've
        #   slept long enough for the drawing thread to go.  (The drawing
        #   thread sets the flag that this wait() waits for.)
        #
        # This has two implications.  First, rater will never allow you
        #   to have more cycles per second than the vblank of your
        #   monitor (if that's what the OpenGL drawing is synced to).
        #   So, if that's 60Hz, any rate call with a number greater than
        #   60 will act as if it were only 60.  Second, if your drawing
        #   thread takes up time close to or greater than the period
        #   you request from .rate(), there may be a period interaction
        #   between the frequency .rate() is really called and the
        #   frequency at which drawing can happen that means we'll spend
        #   some fraction of the cycle waiting here even though we're
        #   not sleeping at all above.
        #
        self.wait()
        self.clear()
        self._time_of_last_rate_call = time.perf_counter()

