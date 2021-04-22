import json
import os
import logging
import argparse
import multiprocessing as mp
from citlab_python_util.parser.xml.page.page import Page


def generate_prediction_json(xml_files, json_path):
    json_dict = {}

    for xml_file in xml_files:
        # load the page xml file
        page_file = Page(xml_file)
        page_name = os.path.basename(xml_file)
        logging.info(f"Processing {xml_file}")

        # load text regions
        try:
            list_of_txt_regions = page_file.get_text_regions()
        except KeyError:
            list_of_txt_regions = []

        txt_region_dict_list = []

        for txt_region in list_of_txt_regions:
            txt_region_txt = ""

            for line in txt_region.text_lines:
                txt_region_txt += line.text + "\n"

            txt_region_dict_list.append({"text_block_id": txt_region.id, "text": txt_region_txt})

        json_dict.update({page_name: txt_region_dict_list})

    json_object = json.dumps(json_dict, indent=None, ensure_ascii=False)

    # writing to json file
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
        logging.info(f"Dumped json {json_path}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_paths", type=str, help="list file containing paths to pageXML files", required=True)
    parser.add_argument("--json_path", type=str, help="output path for json file", required=True)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of partitions to create from original list file and to compute in parallel")
    args = parser.parse_args()

    xml_files = [line.rstrip('\n') for line in open(args.page_paths, "r")]
    n = args.num_workers
    if n > 1:
        split = (len(xml_files) // n) + 1
        processes = []
        for index, sublist in enumerate([xml_files[i:i + split] for i in range(0, len(xml_files), split)]):
            # generate prediction json for sublist
            json_name = os.path.splitext(os.path.basename(args.json_path))[0]
            json_dir = os.path.dirname(args.json_path)
            json_path = os.path.join(json_dir, json_name + "_" + str(index) + ".json")
            # start worker
            p = mp.Process(target=generate_prediction_json, args=(sublist, json_path))
            p.start()
            logging.info(f"Started worker {index}")
            processes.append(p)
            # save sublist to file
            sublist_path = os.path.join(json_dir, json_name + "_" + str(index) + ".lst")
            with open(sublist_path, "w") as lst_file:
                for path in sublist:
                    lst_file.write(path + "\n")
                logging.info(f"Wrote sublist {sublist_path}")

        for p in processes:
            p.join()

    else:
        generate_prediction_json(xml_files, args.json_path)
