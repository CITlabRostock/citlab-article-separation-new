import argparse
import os
import logging

from citlab_python_util.parser.xml.page.page import Page
import citlab_python_util.parser.xml.page.page_constants as page_constants


def get_page_stats(path_to_pagexml, region_stats=True, text_line_stats=True, article_stats=True):
    """ Extraction of the information contained by a given Page xml file.

    :param path_to_pagexml: file path of the Page xml
    :return: list of polygons, list of article ids, image resolution
    """
    # load the page xml file
    print(f"Processing {path_to_pagexml}")
    page_file = Page(path_to_pagexml)
    width, height = page_file.get_image_resolution()
    print(f"- Image resolution: width={width}, height={height}")

    # get regions
    dict_of_regions = page_file.get_regions()
    if region_stats:
        for key in dict_of_regions:
            regions = dict_of_regions[key]
            if text_line_stats and key == page_constants.sTEXTREGION:
                text_lines = []
                for text_region in regions:
                    text_lines.extend(text_region.text_lines)
                print(f"- Number of {key}: {len(regions)}, number of text_lines: {len(text_lines)}")
            else:
                print(f"- Number of {key}: {len(dict_of_regions[key])}")

    if article_stats:
        article_dict = page_file.get_article_dict()
        print(f"- Number of articles: {len(set(article_dict.keys()))}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pagexml_list', help="Input list with paths to pagexml files", required=True)
    parser.add_argument('--region_stats', type=bool, default=True, metavar="BOOL",
                        help="Get region stats or not.")
    parser.add_argument('--text_line_stats', type=bool, default=True, metavar="BOOL",
                        help="Get text_line stats or not.")
    parser.add_argument('--article_stats', type=bool, default=True, metavar="BOOL",
                        help="Get article stats or not.")
    args = parser.parse_args()

    with open(args.pagexml_list, "r") as list_file:
        for path in list_file:
            get_page_stats(path.rstrip())
