import argparse
import os

from python_util.parser.xml.page.page import Page

from typing import List


def create_text_file_from_page(page: Page, path_to_save_file=None):
    """
    Given a PAGE-XML document ```page```, write the text of its text lines into a file given by ```path_to_save_file```. Organize the
    text by articles.

    :param page: Given Page object from which the text is to be extracted article-wise.
    :type page: Page
    :param path_to_save_file: Path to the text file where the text should be stored.
    :type path_to_save_file: str
    :return:
    """
    article_dict = page.get_article_dict()
    with open(path_to_save_file, 'w') as f:
        for i, textlines in enumerate(article_dict.values()):
            for tl in textlines:
                if tl.text:
                    f.write(tl.text + "\n")
            if i != len(article_dict) - 1:
                f.write('\n' + '#' * 100 + '\n\n')


def create_text_files_from_page_list(page_list, path_to_save_folder=None):
    """
    Given a list of PAGE-XML documents ```page_list```, write the text of their text lines into different files inside the
    ```path_to_save_folder```. Organize the text by articles.

    :param page_list: Given list of PAGE-XML paths from which the text is to be extracted article-wise.
    :type page_list: List[str]
    :param path_to_save_folder: Path to the folder where the text should be stored inside text files.
    :type path_to_save_folder: str
    :return:
    """
    for page in page_list:
        page_file_name = os.path.basename(page)
        if path_to_save_folder:
            path_to_save_file = os.path.join(path_to_save_folder, page_file_name + '.txt')
        else:
            path_to_save_file = page + ".txt"
        page = Page(page)
        create_text_file_from_page(page, path_to_save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_page_folder', help="Path to a folder only holding PAGE XML files.",
                        default='')
    parser.add_argument('--path_to_page_list', help="Path to a list file holding page paths.",
                        default='')
    parser.add_argument('--path_to_page_file', help="Path to a page file the corresponding text file should be "
                                                    "created of.")
    parser.add_argument('--path_to_save_folder', help="Path to the folder all the date should be saved to. If not given"
                                                      "save the file next to the page file.",
                        default='')

    args = parser.parse_args()

    path_to_page_folder = args.path_to_page_folder
    path_to_page_list = args.path_to_page_list
    path_to_page_file = args.path_to_page_file
    path_to_save_folder = args.path_to_save_folder

    if path_to_page_folder:
        paths_to_page_files = [os.path.join(path_to_page_folder, f) for f in os.listdir(path_to_page_folder) if
                     os.path.isfile(os.path.join(path_to_page_folder, f))]
        create_text_files_from_page_list(paths_to_page_files, path_to_save_folder)
    elif path_to_page_list:
        with open(path_to_page_list, 'r') as pl:
            create_text_files_from_page_list([l.rstrip() for l in pl.readlines()], path_to_save_folder)
    elif path_to_page_file:
        create_text_files_from_page_list([path_to_page_file], path_to_save_folder)
    else:
        print("Please provide one of the three possibilities:\n"
              "\tA path to a page folder.\n"
              "\tA path to a page list file.\n"
              "\tA path to a page file.")
        exit(1)
