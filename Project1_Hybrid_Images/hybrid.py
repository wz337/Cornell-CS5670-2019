import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np


def apply_cross_correlation_2d(channel, kernel):
    '''Given a kernel of arbitrary m * n dimensions, with both m and n being odd,
    compute the cross correlation of the given channel of an image with the given
    kernel, such that the output is of the same dimesnsions as the channel and that
    I assume the pixels out of the channel to be zero.

    Inputs:
        channel:   An channel (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return a channel of the same dimensions as the input channel
        (same width and height)
    '''
    height, width = channel.shape
    kernel_height, kernel_width = kernel.shape

    # pad the channel based on the given kernel
    height_to_pad, width_to_pad = kernel_height / 2, kernel_width / 2
    padded_channel = np.pad(
        channel,
        ((height_to_pad, height_to_pad), (width_to_pad, width_to_pad)),
        'constant'
    )

    # initialize the new_channel to be the same dimension of the given one
    new_channel = np.zeros((height, width))

    # perform cross correlation on the given channel
    for i in range(height):
        for j in range(width):
            new_channel[i][j] = np.sum(
                padded_channel[i : i + kernel_height, j : j + kernel_width] * kernel
            )

    return new_channel


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    #1 slice/not slice
    #2 pad
    #3 apply cross_correlation_2d on each channel
    #4 stack/not stack
    #5 return

    # to check whether a given image is RGB or a grayscale
    if len(img.shape) == 2:  # greyscale image
        return apply_cross_correlation_2d(img, kernel)

    elif len(img.shape) == 3: # RGB image
        channels = []
        height, width, num_of_channels = img.shape

        # apply cross correlation on each channel of the RGB image
        for i in range(3):
            channel = apply_cross_correlation_2d(img[:,:,i], kernel)
            reshaped_channel = channel.reshape((height, width, 1))
            channels.append(reshaped_channel)

        # stack the result from each channel back to form an image
        new_img = np.stack(tuple(channels), axis=-1)
        reshaped_new_img = new_img.reshape((height, width, num_of_channels))

        return reshaped_new_img


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # flip the kernel so we could simply perform cross_correlation_2d to
    # get the convolution result
    return cross_correlation_2d(img, kernel[::-1, ::-1])


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # initialize x^2 + y^2 in the Gaussian equation
    squares = np.zeros((height, width))

    # populate the squares terms of the Gaussian equation
    for h in range(-height/2, height/2 + 1, 1):
        for w in range(-width/2, width/2 + 1, 1):
            # add the offset back to the index so the index starts from 0
            squares[h + height/2][w + width/2] = h**2 + w**2

    # calculate the kernel
    constant = 1 / (2 * np.pi * np.power(sigma, 2))
    kernel = constant * np.exp ( -squares / (2 * np.power(sigma, 2)))

    # return normalized kernel
    return kernel/np.sum(kernel)


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    return np.subtract(img, low_pass(img, sigma, size))

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *=  mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
