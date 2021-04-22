import cv2
import numpy as np
from matplotlib import pyplot as plt
from citlab_python_util.geometry import rectangle as rect


def get_binarization(image_path, show_binarized_image=False):
    """ Computes for a given image the binarized version. In the resulting matrix are black pixels marked with 1 and
        white ones with 0.

    :param image_path: given image
    :param show_binarized_image: bool to decide whether the binarized image is shown or not
    :return: 0-1-matrix with dimension corresponding to the resolution of the given image
    """
    image = cv2.imread(image_path, 0)
    if image is None:
        exit("Problems while loading the image file '{}'.".format(image_path))

    # otsu binarization: returns 0-1-matrix with dimension corresponding to the resolution of the image
    # 0 marks black pixels, 1 marks white pixels
    ret, image_binarized = cv2.threshold(image, 0, 1, cv2.THRESH_OTSU)

    # 0 marks white pixels, 1 marks black pixels
    image_binarized_inv = np.ones(shape=image_binarized.shape, dtype=int) - image_binarized

    if show_binarized_image:
        plt.imshow(image_binarized, 'gray')
        plt.show()

    return image_binarized_inv


def is_whitespace(binarized_image, rectangle, threshold=0.05):
    """ Decides whether a given region of a binarized image represents a whitespace or not.

    :param binarized_image: binarized image as numpy.ndarray, black pixels marked with 1 and white ones with 0
    :param rectangle: rectangle describing the region to be investigated
    :param threshold: threshold to decide whether the observed region is a whitespace or not
    :return: True or False
    """
    sum_of_black_pixels = 0

    # convention: rectangle with height 0 is a line, therefore the loop runs at least one time
    for i in range(rectangle.y, rectangle.y + rectangle.height + 1):
        for j in range(rectangle.x, rectangle.x + rectangle.width + 1):
            sum_of_black_pixels += binarized_image[i, j]

    number_of_pixels_rectangle = (rectangle.height + 1) * (rectangle.width + 1)

    if sum_of_black_pixels / number_of_pixels_rectangle < threshold:
        return True

    return False
