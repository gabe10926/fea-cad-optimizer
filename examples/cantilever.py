"""Rectangular cantilever: the build123d twin of parametric_fixture.FCStd.

Fixed on the x-min face, loaded on the x-max face (default profile: 1000 N in -y).
"""

from build123d import Align, Box

BOUNDS = {"Length": (50, 150), "Width": (10, 30), "Height": (5, 15)}


def part(Length=100.0, Width=20.0, Height=10.0):
    return Box(Length, Width, Height, align=(Align.MIN, Align.MIN, Align.MIN))
