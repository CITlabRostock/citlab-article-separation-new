import cv2
import numpy as np


def apply_transform(img, transform_type=None, kernel_size=(4, 4), kernel_type='rect', iterations=1):
    """Applies a morphological transformation of type `transform_type` with a kernel of size `kernel_size` and type
    `kernel_type` to the image given by `img` (given as numpy_array) and returns the transformed image with the same
    type as the input. Transformations to choose from are:

    ['erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat']

    Kernel types to choose from are:

    ['rect', 'ellipse', 'cross']

    :param kernel_type: specify the kernel type, choose one of ['rect', 'ellipse', 'cross']
    :type kernel_type: str
    :param transform_type: specify the transformation, choose one of ['erosion', 'dilation', 'opening', 'closing', 'gradient', 'tophat', 'blackhat']
    :type transform_type: str
    :param img: input image to perform dilation on
    :type img: np.ndarray
    :param kernel_size: size of the kernel filter
    :type kernel_size: Tuple(int,int)
    :return: the transformed image
    """
    transform_dict = {'erosion': cv2.MORPH_ERODE, 'dilation': cv2.MORPH_DILATE, 'opening': cv2.MORPH_OPEN,
                      'closing': cv2.MORPH_CLOSE, 'gradient': cv2.MORPH_GRADIENT, 'tophat': cv2.MORPH_TOPHAT,
                      'blackhat': cv2.MORPH_BLACKHAT
                      }
    kernel_dict = {'rect': cv2.MORPH_RECT, 'ellipse': cv2.MORPH_ELLIPSE, 'cross': cv2.MORPH_CROSS}

    if kernel_size == (0, 0) or transform_type is None:
        print('Specify a kernel size and a transformation type to apply the transformation. Returning the image as is.')
        return img
    _transform_type = transform_dict[transform_type]
    _kernel_type = kernel_dict[kernel_type]
    kernel = cv2.getStructuringElement(_kernel_type, kernel_size)

    return cv2.morphologyEx(img, _transform_type, kernel, iterations=iterations)


if __name__ == '__main__':
    a = np.array([[0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 1, 0, 0]], np.uint8)

    a_dilation = apply_transform(a, 'dilation', (2, 2), 'rect')
    a_erosion = apply_transform(a, 'erosion', (2, 2), 'rect')
    a_gradient = apply_transform(a, 'gradient', (2, 2), 'rect')
    print(a)
    print(a_dilation)
    print(a_erosion)
    print(a_gradient)
