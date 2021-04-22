import argparse
import json
import os
import re

import matplotlib.pyplot as plt

from citlab_python_util.geometry.polygon import string_to_poly
from citlab_python_util.parser.xml.page import plot
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.plot import COLORS, DEFAULT_COLOR


def list_img_intersect(l1, l2):
    # check intersection over images
    # val contains [img, bl, line_id]
    img1 = [val[0] for val in l1]
    img2 = [val[0] for val in l2]
    img_intersect = [t for t in img1 if t in img2]
    # return intersection over triples
    l1_intersect = [val for val in l1 if val[0] in img_intersect]
    l2_intersect = [val for val in l2 if val[0] in img_intersect]
    res = l1_intersect + l2_intersect
    return res


def get_kws_from_query(js, query):
    matched_kws = []
    for kw in js:
        if re.match(kw, query.upper()):
            matched_kws.append(kw)
    return matched_kws


def get_imgs_from_kw(js, kw):
    image_list = []
    for pos in js[kw]:
        bl = pos["bl"]
        line = pos["line"]
        conf = float(pos["conf"])
        image = pos["image"]
        image = re.sub(r"/storage", "", image)
        image = re.sub(r"/container.bin", "", image)
        image = get_img_filename(image)
        image_list.append((image, bl, line, conf))
    return image_list


def get_img_filename(path: str) -> str:
    img_filename = os.path.basename(path)
    if not img_filename.endswith(('.jpg', '.png', '.tif')):
        raise ValueError(f"Expected an image with a valid extension, but got '{img_filename}' instead.")
    return img_filename


def get_corresponding_page_path(img_path):
    img_filename = os.path.basename(img_path)
    path_to_folder = os.path.dirname(img_path)

    return os.path.join(path_to_folder, "page", os.path.splitext(img_filename)[0] + ".xml")


def get_textline_by_id(textlines, id):
    for textline in textlines:
        if textline.id == id:
            return textline

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_folder', help="Path to the folder where the image data and the json file is stored."
                                                 "This is also the output folder.")
    parser.add_argument('--json_file', help="Name of json file with kws results, must be stored in --path_to_folder.")
    parser.add_argument('--query_file', help="Text file with one query per line.")
    parser.add_argument('--query', nargs='+', help="Keyword pattern to match. Supports AND/OR. E.g. 'Hello AND World'")
    args = parser.parse_args()

    path_to_json = os.path.join(args.path_to_folder, args.json_file)
    path_to_image_folder = os.path.join(args.path_to_folder, "images")
    path_to_query = os.path.join(args.path_to_folder, args.query_file)

    image_paths = []

    for dirpath, _, filenames in os.walk(path_to_image_folder):
        image_paths.extend([os.path.join(dirpath, f) for f in filenames if f.endswith(('.jpg', '.png', '.tif'))])

    with open(path_to_query, "r") as query_file:
        queries = [q.rstrip() for q in query_file.readlines()]

    with open(path_to_json, "r") as json_file:
        # load json file
        js = json.load(json_file)
        # print(js['keywords'][0]['pos'][0]['conf'])
        # exit(1)
        # print(json.dumps(js, indent=4, sort_keys=True))

        # extract keywords and corresponding matches
        kws_results = {}
        for kw in js["keywords"]:
            kws_results[kw["kw"]] = kw["pos"]

        for _query in queries:
            # Analyze query
            query_list = []
            # for query in args.query:
            for query in _query.split():
                # Look at non-special queries
                if query not in ('AND', 'OR', '(', ')'):
                    query_img_matches = []
                    # Get the matching keywords from the json
                    # matched_kws = get_kws_from_query(kws_results, query)
                    matched_kws = [query.upper()]
                    for kw in matched_kws:
                        # Get the corresponding image results from the json
                        query_img_matches += get_imgs_from_kw(kws_results, kw)
                    # print(f"{query:10} matches {kw:20} -- {query_img_matches[query]}")
                    query_list.append(query_img_matches)
                else:
                    query_list.append(query)

            # Evaluate boolean expression in query iteratively
            # The query is a list containing items and special expressions (booleans, brackets etc.)
            # TODO: How to evaluate brackets? (find last opening bracket + first closing bracket -> eval inbetween)
            while len(query_list) > 2:
                # Evaluate triple aRb (this only works when items and expressions alternate,
                # i.e. just OR/AND, no brackets/NOT, e.g. aRbRcRd...)
                sub_list = query_list[-3:]
                # print(sub_list)
                if sub_list[1].upper() == 'AND':
                    eval_list = list_img_intersect(sub_list[0], sub_list[2])
                elif sub_list[1].upper() == 'OR':
                    eval_list = list(set(sub_list[0] + sub_list[2]))
                else:
                    raise ValueError(f"Unknown keyword {sub_list[1]}")
                # Substitute triple aRb in query with the evaluation of it
                query_list = query_list[:-3]
                query_list.append(eval_list)
            # Result is a single list containing the images & baselines
            query_results = query_list[0]

            relevant_images = set([qr[0] for qr in query_results])

            path_to_query_folder = os.path.join(args.path_to_folder, "queries", _query)
            if not os.path.exists(path_to_query_folder):
                os.makedirs(path_to_query_folder)
            path_to_info_file = os.path.join(path_to_query_folder, "info.txt")

            with open(path_to_info_file, 'w') as info_file:
                info_file.write(f"Number of hits: {len(query_results)}\n")
                info_file.write(f"Number of relevant images: {len(relevant_images)}\n")
                info_file.write(f"Query '{_query}' results in the following matches:\n")
                info_file.write(json.dumps(query_results, indent=2))

            # print(f"\nQuery '{_query}' results in the following matches:")
            # print(f"Number of hits: {len(query_results)}")
            # print(f"Number of relevant images: {len(relevant_images)}\n")
            # print(json.dumps(query_results, indent=2))

            result_images_paths = []
            for dirpath, _, filenames in os.walk(path_to_query_folder):
                result_images_paths.extend(
                    [os.path.join(dirpath, f) for f in filenames if f.endswith(('.jpg', '.png', '.tif'))])

            # Create relevant images with marked hits
            for curr_img in relevant_images:
                # Check if files for curr_img already exit
                skip = False
                for result_image_path in result_images_paths:
                    if curr_img in result_image_path:
                        print(f'Skipping image {curr_img} since the result files for query {_query} already exist.')
                        skip = True
                if skip:
                    continue

                # Get full path
                for image_path in image_paths:
                    if curr_img in image_path:
                        curr_img_path = image_path
                        break
                # print(curr_img_path)

                # open corresponding PAGE file
                page_path = get_corresponding_page_path(curr_img_path)
                page = Page(page_path)
                article_dict = page.get_article_dict()
                used_article_ids = []

                fig, ax = plt.subplots()
                plot.add_image(ax, curr_img_path)

                highest_conf_img = 0.0

                with open(os.path.join(path_to_query_folder, curr_img + ".txt"), 'w+') as text_file:
                    text_file.write(f"QUERY: '{_query}'\n\n")


                for hit in query_results:
                    if hit[0] == curr_img:
                        line_id = hit[2]
                        baseline_hit = hit[1]  # e.g. 1765,4884 3166,4878
                        conf = hit[3]

                        if highest_conf_img < conf:
                            highest_conf_img = conf

                        for aid, textlines in article_dict.items():
                            relevant_textline = get_textline_by_id(textlines, line_id)
                            if relevant_textline is None:
                                continue
                            if aid in used_article_ids:
                                # Just plot the baseline of the hit
                                plot.add_polygons(ax, [string_to_poly(baseline_hit.replace(" ", ";"))],
                                                  color="red", linewidth=0.5, alpha=1)
                                break
                            # Write Text of Baseline Cluster to the text file
                            with open(os.path.join(path_to_query_folder, curr_img + ".txt"), 'a') as text_file:
                                text_file.write(f"HIT FOR TEXTLINE WITH TEXT '{relevant_textline.text}' AND ID"
                                                f" '{relevant_textline.id}' WITH A CONFIDENCE VALUE OF {conf}.\n")
                                text_file.write(f"CORRESPONDING BASELINE CLUSTER:\n\n")
                                for textline in textlines:
                                    if textline.text:
                                        text_file.write(textline.text + "\n")
                                text_file.write("\n#########################\n\n")
                            # Plot baselines in cluster (with same article id)
                            # Create polylist from textlines list
                            poly_list = [tl.baseline.points_list for tl in textlines]
                            plot.add_polygons(ax, poly_list, color=COLORS[len(used_article_ids)], alpha=0.2)
                            used_article_ids.append(aid)
                            plot.add_polygons(ax, [string_to_poly(baseline_hit.replace(" ", ";"))], color="red",
                                              linewidth=0.2, alpha=1)

                conf_as_str = str(highest_conf_img % 1)[2:10]
                img_name_conf = conf_as_str + "_" + curr_img
                text_name_conf = conf_as_str + "_" + curr_img + ".txt"

                plt.axis('off')
                fig.savefig(os.path.join(path_to_query_folder, img_name_conf), bbox_inches='tight', pad_inches=0,
                            dpi=1000)

                fig.clear()
                plt.close(fig)

                os.rename(os.path.join(path_to_query_folder, curr_img + ".txt"),
                          os.path.join(path_to_query_folder, text_name_conf))
