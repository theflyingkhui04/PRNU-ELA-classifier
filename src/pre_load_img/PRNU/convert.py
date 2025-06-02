from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import math
from scipy.signal import convolve2d 

from skimage.transform import resize
import random


def rgb2gray(image: np.ndarray):
    gray_conversion_weights = np.array([0.29893602, 0.58704307, 0.11402090], dtype=np.float32)
    weighted_vector = gray_conversion_weights.reshape((3, 1))

    if image.ndim == 2:
        grayscale_image = image.astype(np.float32)
    elif image.ndim == 3:
        num_channels = image.shape[2]
        if num_channels == 1:
            grayscale_image = image[:, :, 0].astype(np.float32)
        elif num_channels == 3:
            height, width = image.shape[:2]
            pixels = image.reshape((height * width, 3))
            grayscale_values = np.dot(pixels, weighted_vector)
            grayscale_image = grayscale_values.reshape((height, width))
        elif num_channels == 4: 
            image = image[:, :, :3]
            height, width = image.shape[:2]
            pixels = image.reshape((height * width, 3))
            grayscale_values = np.dot(pixels, weighted_vector)
            grayscale_image = grayscale_values.reshape((height, width))
        else:
            raise ValueError('Phải là ảnh xám hoặc ảnh màu')
    else:
        raise ValueError('Phải là ảnh xám hoặc ảnh màu')

    return grayscale_image.astype(np.float32)

def generate_gaussian_kernel( sigma, ksize=None):
    if ksize is None:
        ksize = int(6 * sigma+1) 
        if ksize % 2 == 0:
            ksize += 1

    ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    kernel = kernel / np.sum(kernel)
    return kernel


def gaussian_filter(image, sigma=1.5):
    kernel = generate_gaussian_kernel(sigma)
    if image.ndim == 2:
        return convolve2d(image, kernel, mode='same', boundary='symm')
        
    elif image.ndim == 3:
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
        return blurred

    else:
        raise ValueError("Input ảnh phải là ảnh 2D (grayscale) hoặc 3D (RGB).")


def zero_mean(im: np.ndarray):
    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    ch_mean = im.mean(axis=0).mean(axis=0)
    ch_mean.shape = (1, 1, ch)
    i_zm = im - ch_mean

    row_mean = i_zm.mean(axis=1)
    col_mean = i_zm.mean(axis=0)

    row_mean.shape = (h, 1, ch)
    col_mean.shape = (1, w, ch)

    i_zm_r = i_zm - row_mean
    i_zm_rc = i_zm_r - col_mean

    if im.shape[2] == 1:
        i_zm_rc.shape = im.shape[:2]

    return i_zm_rc

def zero_mean_total(im: np.ndarray):
    im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
    im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
    im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
    im[1::2, 1::2] = zero_mean(im[1::2, 1::2])
    return im



noise_shape = (224, 224)
def extract_single(img, sigma):
    a = rgb2gray(img.astype(np.float32))
    b = gaussian_filter(a, sigma=sigma)
    noise = a - b

    if noise.ndim == 3:
        noise = rgb2gray(noise)

    noise = zero_mean_total(noise)

    resized_noise = resize(noise, noise_shape, preserve_range=True, anti_aliasing=True)
    return resized_noise

def freqq(noise):
    fft = np.fft.fft2(noise) #chuyen noise ve dang song o so phuc
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)
    mag_log = np.log1p(mag)
    mag_norm = (mag_log - np.min(mag_log)) / (np.max(mag_log) - np.min(mag_log))
    return mag_norm

def consis_map(noise, window_size):
    h, w = noise.shape
    heat_map = np.zeros_like(noise)
    glob_std = np.std(noise)

    for i in range(0, h-window_size, window_size//2): #buoc nhay nho
        for j in range(0, w-window_size, window_size//2):
            patch = noise[i:i+window_size, j:j+window_size]
            local_std = np.std(patch)
            consis =np.abs(local_std - glob_std) / (glob_std + 1e-8)
            heat_map[i:i+window_size, j:j+window_size] = consis
    heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))
    return heat_map


