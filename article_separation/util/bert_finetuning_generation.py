import json
import os
import argparse
import logging
import numpy as np
from citlab_python_util.parser.xml.page.page import Page


def generate_finetuning_json(page_paths, json_path):
    xml_files = [line.rstrip('\n') for line in open(page_paths, "r")]
    json_dict = {"page": []}

    region_skips = 0
    page_skips = 0
    for xml_file in xml_files:
        # load the page xml files
        page_file = Page(xml_file)
        page_name = os.path.splitext(os.path.basename(xml_file))[0]
        logging.info(f"Processing {xml_file}")

        # load text regions
        list_of_text_regions = page_file.get_text_regions()

        article_id_text_region_id_dict = {}
        text_region_id_text_lines_dict = {}

        for text_region in list_of_text_regions:
            text_region_article_ids = []
            for text_line in text_region.text_lines:
                if text_line.get_article_id() is not None:
                    text_region_article_ids.append(text_line.get_article_id())
            if not text_region_article_ids:
                logging.warning(f"{xml_file} - {text_region.id} - contains no article_IDs. Skipping.")
                continue
            values, counts = np.unique(text_region_article_ids, return_counts=True)
            index = np.argmax(counts)
            if len(values) > 1:
                logging.warning(f"{xml_file} - {text_region.id} - contains multiple article IDs "
                                f"({set(text_region_article_ids)}). Choosing maximum occurence ({values[index]}).")

            # article id of the text region
            article_id = values[index]
            if article_id not in article_id_text_region_id_dict:
                article_id_text_region_id_dict.update({article_id: [text_region.id]})
            else:
                article_id_text_region_id_dict[article_id].append(text_region.id)
            text_region_id_text_lines_dict.update({text_region.id: text_region.text_lines})

        article_list_of_dicts = []
        for article_id in article_id_text_region_id_dict:
            text_region_list_of_dicts = []

            for text_region_id in article_id_text_region_id_dict[article_id]:
                text_region_txt = ""

                for line in text_region_id_text_lines_dict[text_region_id]:
                    text_region_txt += line.text + "\n"

                text_region_list_of_dicts.append({"text_block_id": text_region_id, "text": text_region_txt})

            article_list_of_dicts.append({"article_id": article_id, "text_blocks": text_region_list_of_dicts})

        json_dict["page"].append({"page_file": page_name, "articles": article_list_of_dicts})

    json_object = json.dumps(json_dict, ensure_ascii=False, indent=None)

    # writing to json file
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
        logging.info(f"Dumped json {json_path}")
    logging.info(f"Number of region skips = {region_skips}")
    logging.info(f"Number of page skips = {page_skips}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_paths", type=str, help="list file containing paths to pageXML files for GT generation",
                        required=True)
    parser.add_argument("--json_path", type=str, help="output path for GT json file", required=True)
    args = parser.parse_args()

    generate_finetuning_json(args.page_paths, args.json_path)
