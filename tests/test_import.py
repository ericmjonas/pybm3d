#!/usr/bin/env python
# coding=utf-8

import pytest


def test_import():
    import pybm3d
    assert pybm3d.bm3d.hello() == "Hello World"

