#!/usr/bin/env python
# coding=utf-8
"""Tests for package imports."""


def test_import():
    """Tests for BM3D function availability."""
    import pybm3d
    assert callable(pybm3d.bm3d.bm3d)
