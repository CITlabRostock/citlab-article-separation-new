# -*- coding: utf-8 -*-

import subprocess
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool


def worker(sample, counter, flags, skipped_files):
    my_tool_subprocess = \
        subprocess.Popen(["python", "citlab_article_separation/baseline_clustering/baseline_clustering.py",
                          "--path_to_xml_file", sample,
                          "--min_polygons_for_cluster", str(flags.min_polygons_for_cluster),
                          "--min_polygons_for_article", str(flags.min_polygons_for_article),
                          "--rectangle_interline_factor", str(flags.rectangle_interline_factor),
                          "--des_dist", str(flags.des_dist),
                          "--max_d", str(flags.max_d),
                          "--use_java_code", str(flags.use_java_code),
                          "--target_average_interline_distance", str(flags.target_average_interline_distance)],
                         shell=False, stdout=subprocess.PIPE)

    # get the print information of the baseline_clustering.py code
    line = True
    outputs = []

    while line:
        line = my_tool_subprocess.stdout.readline()
        outputs.append(line.decode("utf-8"))

    outputs = [line.rstrip('\n') for line in outputs]

    print("No {:5d}: {}".format(counter, sample))
    print("Number of (detected) baselines contained by the image: {}".
          format(outputs[1].split(" ")[-1]))
    print("Number of detected articles (inclusive the \"noise\" class): {}\n".
          format(outputs[2].split(" ")[-1]))

    # saving error when exists
    if outputs[3] != '':
        print(outputs[3])
        skipped_files.append(outputs[3])


if __name__ == "__main__":
    parser = ArgumentParser()
    # command-line arguments
    parser.add_argument('--path_to_xml_lst', type=str, required=True,
                        help="path to the lst file containing the file paths of the page xml's to be processed")

    parser.add_argument('--min_polygons_for_cluster', type=int, default=2,
                        help="minimum number of required polygons in neighborhood to form a cluster")
    parser.add_argument('--min_polygons_for_article', type=int, default=1,
                        help="minimum number of required polygons forming an article")

    parser.add_argument('--rectangle_interline_factor', type=float, default=1.25,
                        help="multiplication factor to calculate the height of the rectangles during the clustering "
                             "progress with the help of the interline distances")

    parser.add_argument('--des_dist', type=int, default=5,
                        help="desired distance (measured in pixels) of two adjacent pixels in the normed polygons")
    parser.add_argument('--max_d', type=int, default=500,
                        help="maximum distance (measured in pixels) for the calculation of the interline distances")
    parser.add_argument('--use_java_code', type=bool, default=True,
                        help="usage of methods written in java (faster than python!) or not")
    parser.add_argument('--target_average_interline_distance', type=int, default=50,
                        help="target interline distance for scaling of the polygons")

    parser.add_argument('--num_threads', type=int, default=1,
                        help="number of threads used for the computation")

    flags = parser.parse_args()

    # list of xml file paths
    xml_files = [line.rstrip('\n') for line in open(flags.path_to_xml_lst, "r")]
    # set the number of workers (default is one)
    tpool = ThreadPool(flags.num_threads)

    skipped_files = []
    print("####################\ntotal number of xml files:")
    print(len(xml_files))
    print("####################\n")

    for xml_counter, xml_file in enumerate(xml_files):
        tpool.apply_async(worker, (xml_file, xml_counter + 1, flags, skipped_files))

    # start the jobs
    tpool.close()
    tpool.join()

    # print possible saving errors
    print("####################\nsaving errors:")
    for skipped_file in skipped_files:
        print(skipped_file)
    print("####################\n")
