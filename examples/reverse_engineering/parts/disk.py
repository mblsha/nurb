from nurb import *


@part
def disk(radius=20.0, height=8.0):
    return Cylinder(radius, height)
