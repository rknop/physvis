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
# Observer pattern

import uuid
import threading

class Subject(object):
    """Subclass this to create something from which Observers will listen for messages.

    Incidentally, each thing that is a Subject has a _id field that is a
    UUID, that you can use to make sure that something is the thing that
    it is.

    Includes _threadlock, a threading.RLock object, you can use for
    global thread locking.

    """

    _threadlock = threading.RLock()

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


