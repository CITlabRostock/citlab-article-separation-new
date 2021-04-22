import logging
from citlab_python_util.parser.xml.page.page import Page
from citlab_article_separation.gnn.input.feature_generation import discard_text_regions_and_lines as discard_regions
from tqdm import tqdm
from argparse import ArgumentParser
import sys

logger = logging.getLogger()
logger.setLevel("DEBUG")

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run(page_path_list, overwrite):
    for page_path in tqdm(page_path_list):
        page = Page(page_path)

        text_regions = page.get_text_regions()

        for text_region in text_regions:
            # text_region_nd = page.get_child_by_id(page.page_doc, text_region.id)
            text_lines = []
            for text_line in text_region.text_lines:
                text_line_nodes = page.get_child_by_id(page.page_doc, text_line.id)
                # duplicate found
                if len(text_line_nodes) > 1:
                    if len(text_line_nodes) >= 3:
                        raise Exception(f"Expected at most two text lines with the same id, but found "
                                        f"{len(text_line_nodes)}.")
                    # check which one is the duplicate (has no text region ancestor)
                    line1 = text_line_nodes[0]
                    line1_has_region = bool(page.get_ancestor_by_name(line1, "TextRegion"))
                    line2 = text_line_nodes[1]
                    line2_has_region = bool(page.get_ancestor_by_name(line2, "TextRegion"))
                    if line1_has_region and not line2_has_region:
                        line = line1
                        duplicate = line2
                    elif line2_has_region and not line1_has_region:
                        line = line2
                        duplicate = line1
                        # set article id only if the duplicate was line1, otherwise the article id was already correct
                        article_id = page.parse_custom_attr(duplicate.get("custom"))["structure"]["id"]
                        text_line.set_article_id(article_id)
                    else:
                        raise Exception(f"Can't correctly determine duplicate text line.")
                    # remove duplicate
                    page.remove_page_xml_node(duplicate)
                # aggregate (partly updated) text lines
                text_lines.append(text_line)
            # overwrite text lines in text region
            page.set_text_lines(text_region, text_lines, overwrite=True)

        # overwrite text regions (do this after the text lines, so we also catch duplicates of text regions that get
        # discarded)
        text_regions, _ = discard_regions(text_regions)
        page.set_text_regions(text_regions, overwrite=True)

        if overwrite:
            page.write_page_xml(page_path)
        else:
            page.write_page_xml(page_path + ".xml")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--page_path_list', default='', type=str, metavar="STR",
                        help="path to the lst file containing the file paths of the PageXmls")
    parser.add_argument('--overwrite', default=False, type=bool,
                        help="If true, it overwrites the page xml files modified by the preprocessor. "
                             "Defaults to False.")

    flags = parser.parse_args()
    page_path_list = flags.page_path_list
    overwrite = flags.overwrite

    with open(page_path_list) as f:
        page_path_list = f.readlines()
    page_path_list = [pp.rstrip() for pp in page_path_list]
    run(page_path_list, overwrite)
