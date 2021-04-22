import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from citlab_python_util.io.file_loader import load_list_file


class ImageBinarizer:
    def __init__(self):
        pass

    def binarize_image(self, image, mode='otsu'):
        if mode == 'thresh':
            return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 16)
        elif mode == 'otsu':
            return cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]

        raise Exception('Please choose a valid binarization mode. Choose from [thresh, otsu].')

    def binarize_images(self, image_list, mode='otsu'):
        return [self.binarize_image(image, mode) for image in image_list]

    def show(self, image):
        plt.imshow(image, cmap='gray')
        plt.show()
        bin_image = self.binarize_image(image, mode='thresh')
        plt.imshow(bin_image, cmap='gray')
        plt.show()


def run_image_binarization(image_path_list, mode='otsu', save_folder=None, keep_folder_depth=0):
    """

    :param image_path_list:
    :param mode:
    :param save_folder:
    :param keep_folder_depth: if 0, put all images in one folder, if 1, put each image in a folder that has the same name the original image was in, ...
    :return:
    """
    ib = ImageBinarizer()

    for image_path in tqdm(image_path_list):
        image_path = Path(image_path)
        if keep_folder_depth < 0:
            raise ValueError(
                f"{keep_folder_depth} is not valid. Please choose a value greater or equal to 0 for keep_folder_depth")
        save_path = Path(save_folder).joinpath(*image_path.parts[-1 - keep_folder_depth:])

        image = cv2.imread(str(image_path.resolve()))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bin_image = ib.binarize_image(image, mode=mode)
        if not save_path.parent.exists():
            save_path.parent.mkdir()
        cv2.imwrite(save_path, bin_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str,
                        help='Path to the file listing the paths to the images to binarize.')
    parser.add_argument('--save_folder', type=str,
                        help='Path to the folder where the images should be stored.')
    parser.add_argument('--keep_folder_depth', type=int,
                        help='Keep the original structure of the image data. 0 means all images are stored in one '
                             'folder independent of their original folder structure. 1 means every image is stored '
                             'into "/path/to/save/folder/parent_folder_of_image_i/image_i.jpg', default=0)
    parser.add_argument('--mode', type=str,
                        help='Which binarization to perform, possible options are "otsu" and "thresh".',
                        default='thresh')

    args = parser.parse_args()

    image_path_list = load_list_file(args.image_list)
    save_folder = args.save_folder
    keep_folder_depth = args.keep_folder_depth
    mode = args.mode

    run_image_binarization(image_path_list, mode=mode, save_folder=save_folder, keep_folder_depth=keep_folder_depth)
