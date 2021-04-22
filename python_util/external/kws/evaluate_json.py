import argparse
import json
import os
import re
from copy import deepcopy

import matplotlib.pyplot as plt

from citlab_python_util.geometry.polygon import string_to_poly, are_vertical_aligned
from citlab_python_util.parser.xml.page import plot
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.plot import COLORS


def are_vertically_close(poly1, poly2, min_dist_x=200, max_dist_x=1750, max_dist_y=100):
    poly1 = string_to_poly(poly1)
    poly2 = string_to_poly(poly2)

    poly1_avg_y = sum(poly1.y_points) / len(poly1.y_points)
    poly2_avg_y = sum(poly2.y_points) / len(poly2.y_points)
    poly1_avg_x = sum(poly1.x_points) / len(poly1.x_points)
    poly2_avg_x = sum(poly2.x_points) / len(poly2.x_points)

    # TODO: FIND OPTIMAL VALUES, maybe based on the average textline length per page
    if abs(poly1_avg_y - poly2_avg_y) < max_dist_y and min_dist_x < abs(
            poly1_avg_x - poly2_avg_x) < max_dist_x and poly1_avg_y < poly2_avg_y and not max(poly1.x_points) < min(
        poly2.y_points):
        return True

    return False


def list_img_intersect_with_textline_cond(l1, l2):
    # val contains [img, bl, line_id, conf]
    # intersect = [(val1, val2) for val1 in l1 for val2 in l2 if
    # (val1[0] == val2[0] and are_vertically_close(val1[1].replace(" ", ";"), val2[1].replace(" ", ";")))]
    intersect = [(val1, val2) for val1 in l1 for val2 in l2 if
                 (val1[0] == val2[0] and are_vertically_close(val1[1].replace(" ", ";"), val2[1].replace(" ", ";")))]

    return intersect


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


def get_hyphenation_results(hyph_dict, keyword):
    try:
        hyph_list = hyph_dict[keyword]
    except KeyError:
        print(f"Found no corresponding entry for {keyword} in the hyphenation file. "
              f"Just search for the keyword itself.")
        hyph_list = []

    for hyph_tuple in hyph_list:
        suffix_results = suffix_kws_result[hyph_tuple[0].upper()]
        if not suffix_results:
            continue
        if hyph_tuple[1]:
            prefix_results = prefix_kws_result[hyph_tuple[1].upper()]
            if not prefix_results:
                continue
        else:
            prefix_results = []

        suffix_imgs_matches = get_imgs_from_kw(suffix_kws_result, hyph_tuple[0].upper())
        if prefix_results:
            prefix_imgs_matches = get_imgs_from_kw(prefix_kws_result, hyph_tuple[1].upper())
            prefix_suffix_intersect = list_img_intersect_with_textline_cond(suffix_imgs_matches,
                                                                            prefix_imgs_matches)
        else:
            prefix_suffix_intersect = suffix_imgs_matches

        return prefix_suffix_intersect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_folder', help="Path to the folder where the image data and the json file is stored."
                                                 "This is also the output folder.")
    parser.add_argument('--kws_result_file',
                        help="Name of json file with kws results, must be stored in --path_to_folder.")
    parser.add_argument('--prefix_file', help="Name of the json file with kws results at the beginning of a line, when"
                                              "a word gets hyphenated.")
    parser.add_argument('--suffix_file', help="Name of the json file with kws results at the end of a line, when"
                                              "a word gets hyphenated.")
    parser.add_argument('--query_file', help="Text file with one query per line.")
    parser.add_argument('--hyphenation_file',
                        help="JSON file holding information to all possible hyphenations of a query")
    parser.add_argument('--only_hyphenations', help="Whether the script should only search for hyphenations, or also"
                                                    "the keyword itself.")
    args = parser.parse_args()

    only_hyphenations = args.only_hyphenations.lower() in ['true', '1', 'yes']

    path_to_prefix_file = None
    path_to_suffix_file = None

    if args.prefix_file:
        path_to_prefix_file = os.path.join(args.path_to_folder, args.prefix_file)
    if args.suffix_file:
        path_to_suffix_file = os.path.join(args.path_to_folder, args.suffix_file)

    path_to_kws_result_file = os.path.join(args.path_to_folder, args.kws_result_file)
    path_to_image_folder = os.path.join(args.path_to_folder, "images")
    path_to_query = os.path.join(args.path_to_folder, args.query_file)
    path_to_hyphenation_file = os.path.join(args.path_to_folder, args.hyphenation_file)

    image_paths = []

    for dirpath, _, filenames in os.walk(path_to_image_folder):
        image_paths.extend([os.path.join(dirpath, f) for f in filenames if f.endswith(('.jpg', '.png', '.tif'))])

    with open(path_to_query, "r") as query_file:
        queries = [q.rstrip() for q in query_file.readlines()]

    with open(path_to_hyphenation_file, "r") as hyph_file:
        hyph_dict = json.load(hyph_file)

    with open(path_to_kws_result_file, "r") as kws_res_file:
        # load json file
        js = json.load(kws_res_file)

        # extract keywords and corresponding matches
        kws_results = {}
        for kw in js["keywords"]:
            kws_results[kw["kw"]] = kw["pos"]

    with open(path_to_prefix_file, "r") as prefix_file:
        prefix_dict = json.load(prefix_file)

        prefix_kws_result = {}
        for kw in prefix_dict["keywords"]:
            prefix_kws_result[kw["kw"]] = kw["pos"]

    with open(path_to_suffix_file, "r") as suffix_file:
        suffix_dict = json.load(suffix_file)

        suffix_kws_result = {}
        for kw in suffix_dict["keywords"]:
            suffix_kws_result[kw["kw"]] = kw["pos"]

    for query in queries:
        # query = "A AND B"
        query_list = []
        for query_part in query.split():
            # query_part iterating over ["A", "AND", "B"]
            # Look at non-special queries
            if query_part.upper() not in ('AND', 'OR', '(', ')'):
                prefix_suffix_intersection = get_hyphenation_results(hyph_dict, query_part)
                query_img_matches = []
                matched_kws = [query_part.upper()]
                for kw in matched_kws:
                    query_img_matches += get_imgs_from_kw(kws_results, kw)
                query_list.append((query_img_matches, prefix_suffix_intersection))
            else:
                query_list.append(query_part)

        while len(query_list) > 2:
            sub_list = query_list[-3:]
            if sub_list[1].upper() == "AND":
                img_list1 = [val[0] for val in sub_list[0][0]] + [val[0][0] for val in sub_list[0][1]]
                img_list2 = [val[0] for val in sub_list[2][0]] + [val[0][0] for val in sub_list[2][1]]
                img_intersect = [img for img in img_list1 if img in img_list2]

                full_query_match_intersect1 = [val for val in sub_list[0][0] if val[0] in img_intersect]
                prefix_suffix_intersect1 = [val for val in sub_list[0][1] if val[0][0] in img_intersect]

                full_query_match_intersect2 = [val for val in sub_list[2][0] if val[0] in img_intersect]
                prefix_suffix_intersect2 = [val for val in sub_list[2][1] if val[0][0] in img_intersect]

                eval_result = (full_query_match_intersect1 + full_query_match_intersect2,
                               prefix_suffix_intersect1 + prefix_suffix_intersect2)
            elif sub_list[1].upper() == "OR":
                eval_result = (list(set(sub_list[0][0] + sub_list[2][0])),
                               list(set(sub_list[0][1] + sub_list[2][1])))
            else:
                raise ValueError(f"Unknown keyword {sub_list[1]}.")
            query_list = query_list[:-3]
            query_list.append(eval_result)
        query_results = query_list[0]

        relevant_images = set([qr[0] for qr in query_results[0]] + [qr[0][0] for qr in query_results[1]])

        path_to_query_folder = os.path.join(args.path_to_folder, "queries_test", query)
        if not os.path.exists(path_to_query_folder):
            os.makedirs(path_to_query_folder)
        path_to_info_file = os.path.join(path_to_query_folder, "info.txt")

        result_images_paths = []
        for dirpath, _, filenames in os.walk(path_to_query_folder):
            result_images_paths.extend(
                [os.path.join(dirpath, f) for f in filenames if f.endswith(('.jpg', '.png', '.tif'))])

        for curr_img in relevant_images:
            skip = False
            for result_image_path in result_images_paths:
                if curr_img in result_image_path:
                    print(f"Skipping image {curr_img} since the result files for the query {query} already exist.")
                    skip = True
                    break
            if skip:
                continue

            curr_img_path = None
            for image_path in image_paths:
                if curr_img in image_path:
                    curr_img_path = image_path
                    break

            if not curr_img_path:
                # raise ValueError(f"Path for image {curr_img} not found.")
                continue

            # open corresponding PAGE file
            page_path = get_corresponding_page_path(curr_img_path)
            page = Page(page_path)
            article_dict = page.get_article_dict()
            used_article_ids = []

            fig, ax = plt.subplots()
            plot.add_image(ax, curr_img_path)

            if " AND " in query.upper():
                conf_img = 1.0
            else:
                conf_img = 0.0

            with open(os.path.join(path_to_query_folder, curr_img + ".txt"), 'w+') as text_file:
                text_file.write(f"QUERY: '{query}'\n\n")

            has_full_hit = False
            skip_img = False
            for full_hit in query_results[0]:
                if full_hit[0] == curr_img:
                    has_full_hit = True
                    line_id = full_hit[2]
                    baseline_hit = full_hit[1]  # e.g. 1765,4884 3166,4878
                    conf = full_hit[3]

                    if " AND " in query.upper():
                        # Take minmal value
                        if conf_img > conf:
                            conf_img = conf
                    else:
                        # Take maximal value
                        if conf_img < conf:
                            conf_img = conf

                    for aid, textlines in article_dict.items():
                        relevant_textline = get_textline_by_id(textlines, line_id)
                        if relevant_textline is None:
                            continue

                        # Write Text of Baseline Cluster to the text file
                        with open(os.path.join(path_to_query_folder, curr_img + ".txt"), 'a') as text_file:
                            text_file.write(f"HIT FOR TEXTLINE WITH TEXT '{relevant_textline.text}' AND ID"
                                            f" '{relevant_textline.id}' WITH A CONFIDENCE VALUE OF {conf}.\n")
                            text_file.write(f"CORRESPONDING BASELINE CLUSTER:\n\n")
                            for textline in textlines:
                                if textline.text:
                                    text_file.write(textline.text + "\n")
                            text_file.write("\n#########################\n\n")

                        if aid in used_article_ids:
                            # Just plot the baseline of the hit
                            plot.add_polygons(ax, [string_to_poly(baseline_hit.replace(" ", ";"))],
                                              color="red", linewidth=0.2, alpha=1)
                            break

                        # Plot baselines in cluster (with same article id)
                        # Create polylist from textlines list
                        poly_list = [tl.baseline.points_list for tl in textlines]
                        plot.add_polygons(ax, poly_list, color=COLORS[len(used_article_ids)], alpha=0.2)
                        used_article_ids.append(aid)
                        plot.add_polygons(ax, [string_to_poly(baseline_hit.replace(" ", ";"))], color="red",
                                          linewidth=0.2, alpha=1)

            for hyph_hit in query_results[1]:
                if hyph_hit[0][0] == curr_img:
                    line_id1 = hyph_hit[0][2]
                    line_id2 = hyph_hit[1][2]
                    baseline_hit1 = hyph_hit[0][1]  # e.g. 1765,4884 3166,4878
                    baseline_hit2 = hyph_hit[1][1]
                    conf1 = hyph_hit[0][3]
                    conf2 = hyph_hit[1][3]

                    conf = (conf1 + conf2) / 2

                    if " AND " in query.upper():
                        # Take minmal value
                        if conf_img > conf:
                            conf_img = conf
                    else:
                        # Take maximal value
                        if conf_img < conf:
                            conf_img = conf

                    relevant_textline1 = None
                    relevant_textline2 = None
                    for aid, textlines in article_dict.items():
                        if relevant_textline1 is None:
                            relevant_textline1 = get_textline_by_id(textlines, line_id1)
                            aid1 = aid
                            textlines1 = textlines
                        if relevant_textline2 is None:
                            relevant_textline2 = get_textline_by_id(textlines, line_id2)
                            aid2 = aid
                            textlines2 = textlines
                        if relevant_textline1 is None or relevant_textline2 is None:
                            continue

                        if not are_vertical_aligned(relevant_textline1.baseline.points_list,
                                                    relevant_textline2.baseline.points_list):
                            new_hyph_results_list = deepcopy(query_results[1])
                            for i, hyph_result in enumerate(query_results[1]):
                                if hyph_result[0][2] == relevant_textline1.id and hyph_result[1][2] == relevant_textline2.id:
                                    # delete from info file
                                    new_hyph_results_list.remove(hyph_result)
                                    if not has_full_hit:
                                        skip_img = True
                                    break
                            query_results = (query_results[0], new_hyph_results_list)
                            break

                        # Write Text of Baseline Cluster to the text file
                        with open(os.path.join(path_to_query_folder, curr_img + ".txt"), 'a') as text_file:
                            text_file.write(f"HIT FOR TEXTLINES WITH TEXTS '{relevant_textline1.text}' - "
                                            f"'{relevant_textline2.text}', IDS  '{relevant_textline1.id}' - "
                                            f"'{relevant_textline2.id}' AND CONFIDENCE VALUES {conf1} - {conf2}.\n"
                                            )
                            text_file.write(f"CORRESPONDING BASELINE CLUSTER:\n\n")
                            if aid1 == aid2:
                                for textline in textlines1:
                                    if textline.text:
                                        text_file.write(textline.text + "\n")
                                text_file.write("\n#########################\n\n")
                            else:
                                for textline in textlines1:
                                    if textline.text:
                                        text_file.write(textline.text + "\n")
                                text_file.write("-" * 10 + "\n")
                                for textline in textlines2:
                                    if textline.text:
                                        text_file.write(textline.text + "\n")
                                text_file.write("\n#########################\n\n")

                        if aid1 in used_article_ids:
                            # Just plot the baseline of the hit
                            plot.add_polygons(ax, [string_to_poly(baseline_hit1.replace(" ", ";"))],
                                              color="red", linewidth=0.2, alpha=1)

                        else:
                            # Plot baselines in cluster (with same article id)
                            # Create polylist from textlines list
                            poly_list = [tl.baseline.points_list for tl in textlines1]
                            plot.add_polygons(ax, poly_list, color=COLORS[len(used_article_ids)], alpha=0.2)
                            used_article_ids.append(aid1)
                            plot.add_polygons(ax, [string_to_poly(baseline_hit1.replace(" ", ";"))], color="red",
                                              linewidth=0.2, alpha=1)

                        if aid2 in used_article_ids:
                            # Just plot the baseline of the hit
                            plot.add_polygons(ax, [string_to_poly(baseline_hit2.replace(" ", ";"))],
                                              color="red", linewidth=0.2, alpha=1)
                        else:
                            # Plot baselines in cluster (with same article id)
                            # Create polylist from textlines list
                            poly_list = [tl.baseline.points_list for tl in textlines2]
                            plot.add_polygons(ax, poly_list, color=COLORS[len(used_article_ids)], alpha=0.2)
                            used_article_ids.append(aid2)
                            plot.add_polygons(ax, [string_to_poly(baseline_hit2.replace(" ", ";"))], color="red",
                                              linewidth=0.2, alpha=1)

                        break
            if skip_img:
                os.remove(os.path.join(path_to_query_folder, curr_img + ".txt"))

            conf_as_str = str(conf_img % 1)[2:10]
            img_name_conf = conf_as_str + "_" + curr_img
            text_name_conf = conf_as_str + "_" + curr_img + ".txt"

            plt.axis('off')
            fig.savefig(os.path.join(path_to_query_folder, img_name_conf), bbox_inches='tight',
                        pad_inches=0,
                        dpi=1000)

            fig.clear()
            plt.close(fig)

            os.rename(os.path.join(path_to_query_folder, curr_img + ".txt"),
                      os.path.join(path_to_query_folder, text_name_conf))

        with open(path_to_info_file, 'w') as info_file:
            info_file.write(f"Number of hits: {len(query_results[0]) + len(query_results[1])}\n")
            info_file.write(f"\tNumber of full hits: {len(query_results[0])}\n")
            info_file.write(f"\tNumber of hyphenation hits: {len(query_results[1])}\n")
            info_file.write(f"Number of relevant images: {len(relevant_images)}\n")
            info_file.write(f"Query '{query}' results in the following matches:\n")
            info_file.write(json.dumps(query_results, indent=2))
