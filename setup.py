#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="vbeam",
    version="1.0.1",
    description="vbeam: a fast and differentiable beamformer",
    author="Magnus Dalen Kvalevåg",
    author_email="magnus.kvalevag@ntnu.no",
    url="https://github.com/magnusdk/vbeam",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyuff_ustb>=2.0.7",
        "scipy",
        "spekk",
    ],
)
