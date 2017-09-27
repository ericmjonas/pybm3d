#!/usr/bin/env python
# coding=utf-8

import pytest
import numpy as np
import skimage.data

import pybm3d

def test_bm3d():
    x  = skimage.data.camera().astype(np.float32)
    x.shape = x.shape[0], x.shape[1], 1 # correct # of chan

    noise = np.random.normal(0, 40, x.shape).astype(np.float32)
    xn = x + noise


    z = pybm3d.bm3d.bm3d(xn, 40.0)

    noise_error = np.sum(np.abs(xn - x))
    recover_error = np.sum(np.abs(z - x))

    assert noise_error > 4*recover_error

