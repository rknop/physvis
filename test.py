#/usr/bin/python3
# -*- coding: utf-8 -*-

import sys

class TestClass:
    def instance_method(self):
        cls = self.__class__
        sys.stderr.write("TestClass.thing = {}\n".format(TestClass.thing))

    @classmethod
    def class_method(cls, thing):
        TestClass.thing = thing


class NextTestClass(TestClass):
    def check_me_out(self):
        self.__class__.thing = "checked out!"

sys.stderr.write("Initialize\n---------\n")        
it1 = TestClass()
it1.class_method("dog")

it2 = TestClass()
it1.instance_method()
it2.instance_method()
sys.stderr.write("\nit2 calls class_method\n--------\n")
it2.class_method("cat")
it1.instance_method()
it2.instance_method()

sys.stderr.write("\ncreate it3\n-----------")
it3 = NextTestClass()
it3.instance_method()
sys.stderr.write("\nit3.checK_me_out\n---------\n")
it3.check_me_out()
it1.instance_method()
it2.instance_method()
it3.instance_method()

sys.stderr.write("\nit3.class_method(\"it3\")\n-------------\n")
it3.class_method("it3!")
it1.instance_method()
it2.instance_method()
it3.instance_method()
