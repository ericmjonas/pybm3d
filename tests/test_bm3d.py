#!/usr/bin/env python
# coding=utf-8
"""Tests for BM3D image denoising."""

import pytest
import numpy as np
import skimage.data

import pybm3d


@pytest.fixture
def noise_data():
    """Provide grayscale data for denoising."""
    noise_std_dev = 40.0
    img = skimage.data.camera().astype(np.float32)

    noise = np.random.normal(0, noise_std_dev, img.shape).astype(np.float32)
    noisy_img = np.clip(img + noise, 0, 255)
    return img, noisy_img, noise_std_dev


@pytest.fixture
def color_noise_data():
    """Provide color data for denoising."""
    noise_std_dev = 40.0
    img = skimage.data.astronaut().astype(np.float32)

    noise = np.random.normal(0, noise_std_dev, img.shape).astype(np.float32)
    noisy_img = np.clip(img + noise, 0, 255)
    return img, noisy_img, noise_std_dev


def test_import():
    """Tests for BM3D function availability."""
    import pybm3d
    assert callable(pybm3d.bm3d.bm3d)


def test_bm3d(noise_data):
    """Tests BM3D grayscale image denoising."""
    img, noisy_img, noise_std_dev = noise_data

    out = pybm3d.bm3d.bm3d(noisy_img, noise_std_dev)
    out = np.clip(out, 0, 255)

    noise_error = np.sum(np.abs(noisy_img - img))
    out_error = np.sum(np.abs(out - img))

    assert noise_error > 4 * out_error


def test_bm3d_color(color_noise_data):
    """Tests BM3D color image denoising."""
    img, noisy_img, noise_std_dev = color_noise_data

    out = pybm3d.bm3d.bm3d(noisy_img, noise_std_dev)
    out = np.clip(out, 0, 255)

    noise_error = np.sum(np.abs(noisy_img - img))
    out_error = np.sum(np.abs(out - img))

    assert noise_error > 2 * out_error


def test_fail_patch_size_param(noise_data):
    """Tests expected failure for wrong patch_size parameter value."""
    img, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d.bm3d(noisy_img, noise_std_dev, patch_size=-1)


def test_fail_tau_2d_hard_param(noise_data):
    """Tests expected failure for wrong tau_2D_hard parameter value."""
    img, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d.bm3d(noisy_img, noise_std_dev, tau_2D_hard="not_supported")


def test_fail_tau_2d_wien_param(noise_data):
    """Tests expected failure for wrong tau_2D_wien parameter value."""
    img, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d.bm3d(noisy_img, noise_std_dev, tau_2D_wien="not_supported")


def test_fail_color_space_param(noise_data):
    """Tests expected failure for wrong tau_2D_wien parameter value."""
    img, noisy_img, noise_std_dev = noise_data

    with pytest.raises(ValueError):
        pybm3d.bm3d.bm3d(noisy_img, noise_std_dev, color_space="not_supported")
