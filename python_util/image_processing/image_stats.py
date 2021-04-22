import os

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation as inter


def get_scaling_factor(image_height, image_width, scaling_factor, fixed_height=None, fixed_width=None):
    if fixed_height is not None and scaling_factor is not None and 0.1 < scaling_factor:
        return scaling_factor * fixed_height / image_height
    if fixed_width is not None and scaling_factor is not None and 0.1 < scaling_factor:
        return scaling_factor * fixed_width / image_width
    if fixed_height:
        return fixed_height / image_height
    if fixed_width:
        return fixed_width / image_width
    if scaling_factor:
        return scaling_factor


def get_image_dimensions(image_path):
    """
    Get width and height of the image given by `image_path`.
    :param image_path: path to the image
    :return: (width, height) of image
    """
    return PIL.Image.open(image_path).size


def get_rotation_angle(image, delta=0.1, limit=2):
    def find_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        hist = np.sum(data, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        return hist, score

    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(image, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    return best_score, best_angle


def get_number_of_pixels(image: np.uint8) -> int:
    """Returns the number of pixels in an image.

    :param image: input image (given as numpy array)
    :return: number of pixels in the image
    """
    return int(np.prod(image.shape))


def get_pixel_histogram(image: np.uint8, plot_histogram: bool = True) -> [int, list]:
    """Given a grayscale image `image` the function prints information regarding the distribution of pixel values.
    Also plots the pixel-histogram together with the image if `plot_histogram` is True. Returns the number of pixels in the image as well as a list
    containing the number of occurences of each pixel value, i.e. the list has length 256.

    :param plot_histogram: whether to plot the histogram or not
    :param image: input image (given as numpy array)
    :return: number of pixels; list of pixel occurences
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    no_of_pixels = get_number_of_pixels(image)

    print(f"Number of pixels: {no_of_pixels}")

    for i, px_value in enumerate(hist):
        px_value_int = int(px_value[0])
        print(f"Pixel value {i:3} occurs: {px_value_int:8} (relative: {round(px_value_int / no_of_pixels * 100, 4)})")

    if plot_histogram:
        plt.subplot(1, 2, 1)
        plt.title("Pixel histogram")
        plt.hist(image.ravel(), 256, [0, 256])
        plt.subplot(1, 2, 2)
        plt.title("Greyscale image")
        plt.imshow(image, cmap="gray")
        plt.show()

    return no_of_pixels, [int(p[0]) for p in hist]


if __name__ == '__main__':
    path_to_image = "/home/max/devel/projects/python/article_separation/data/test_post_processing/textblock/ONB_aze_19110701_004.jpg"
    image_folder_dir = os.path.dirname(path_to_image)
    image_name, image_ext = os.path.splitext(os.path.basename(path_to_image))

    path_to_tb = os.path.join(image_folder_dir, image_name + "_OUT1" + image_ext)

    image = cv2.imread(path_to_tb, 0)

    no_pixel, hist = get_pixel_histogram(image)

    no_pixel = get_number_of_pixels(image)
    print(no_pixel)
