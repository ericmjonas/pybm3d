#!/usr/bin/env python
# coding=utf-8
"""Tests for BM3D image denoising."""

import multiprocessing
import numpy as np
import skimage.data
from skimage.measure import compare_psnr
import pytest

import pybm3d


@pytest.fixture
def noise_data():
    """Provide grayscale data for denoising."""
    noise_std_dev = 40.0
    img = skimage.data.camera()

    noise = np.random.normal(0, noise_std_dev, img.shape).astype(img.dtype)
    noisy_img = np.clip(img + noise, 0, 255)
    return img, noisy_img, noise_std_dev


@pytest.fixture
def color_noise_data():
    """Provide color data for denoising."""
    noise_std_dev = 40.0
    img = skimage.data.astronaut()

    noise = np.random.normal(0, noise_std_dev, img.shape).astype(img.dtype)
    noisy_img = np.clip(img + noise, 0, 255)
    return img, noisy_img, noise_std_dev


def test_import():
    """Tests for BM3D function availability."""
    import pybm3d
    assert callable(pybm3d.bm3d)


def test_bm3d(noise_data):
    """Tests BM3D grayscale image denoising."""
    img, noisy_img, noise_std_dev = noise_data

    out = pybm3d.bm3d(noisy_img, noise_std_dev)

    noise_psnr = compare_psnr(img, noisy_img)
    out_psnr = compare_psnr(img, out)

    assert out_psnr > noise_psnr


def test_bm3d_color(color_noise_data):
    """Tests BM3D color image denoising."""
    img, noisy_img, noise_std_dev = color_noise_data

    out = pybm3d.bm3d(noisy_img, noise_std_dev)

    noise_psnr = compare_psnr(img, noisy_img)
    out_psnr = compare_psnr(img, out)

    assert out_psnr > noise_psnr


def test_fail_patch_size_param(noise_data):
    """Tests expected failure for wrong patch_size parameter value."""
    _, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d(noisy_img, noise_std_dev, patch_size=-1)


def test_fail_tau_2d_hard_param(noise_data):
    """Tests expected failure for wrong tau_2D_hard parameter value."""
    _, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d(noisy_img, noise_std_dev, tau_2D_hard="not_supported")


def test_fail_tau_2d_wien_param(noise_data):
    """Tests expected failure for wrong tau_2D_wien parameter value."""
    _, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d(noisy_img, noise_std_dev, tau_2D_wien="not_supported")


def test_fail_color_space_param(noise_data):
    """Tests expected failure for wrong tau_2D_wien parameter value."""
    _, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d(noisy_img, noise_std_dev, color_space="not_supported")


def test_fail_no_integer_input(noise_data):
    """Tests expected failure for inputs with type Float."""
    _, noisy_img, noise_std_dev = noise_data

    noisy_img = noisy_img.astype(np.float)

    with pytest.raises(TypeError):
        pybm3d.bm3d(noisy_img, noise_std_dev)


def test_fail_wrong_num_channel_input(noise_data):
    """Tests expected failure for inputs with wrong number of channels.

    Allowed number of color channels are 1 or 3.
    """
    _, noisy_img, noise_std_dev = noise_data

    # build 2 channel input
    noisy_img = np.array([noisy_img, noisy_img]).transpose((1, 2, 0))

    with pytest.raises(IndexError):
        pybm3d.bm3d(noisy_img, noise_std_dev)


def test_fail_exceeding_num_threads(noise_data):
    """Tests expected failure for exceeding num_threads parameter.

    The parameter must not be larger than the number of available CPUs.
    """
    _, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        num_threads = multiprocessing.cpu_count() + 1
        pybm3d.bm3d(noisy_img, noise_std_dev, num_threads=num_threads)
