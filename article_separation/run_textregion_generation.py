# -*- coding: utf-8 -*-

import subprocess
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool


def worker(sample, counter, flags, skipped_files):
    my_tool_subprocess = \
        subprocess.Popen(["python", "citlab_article_separation/textregion_generation/textregion_generation.py",
                          "--path_to_xml_file", sample,
                          "--des_dist", str(flags.des_dist),
                          "--max_d", str(flags.max_d),
                          "--alpha", str(flags.alpha),
                          "--use_java_code", str(flags.use_java_code)],
                         shell=False, stdout=subprocess.PIPE)

    # get the print information of the textregion_generation.py code
    line = True
    outputs = []

    while line:
        line = my_tool_subprocess.stdout.readline()
        outputs.append(line.decode("utf-8"))

    outputs = [line.rstrip('\n') for line in outputs]

    print("No {:5d}: {}".format(counter, sample))
    # saving error when exists
    if outputs[-2] != sample and outputs[-2] != "alpha value not suitable -> is increased":
        print(outputs[-2])
        skipped_files.append(outputs[-2])


if __name__ == "__main__":
    parser = ArgumentParser()
    # command-line arguments
    parser.add_argument('--path_to_xml_lst', type=str, required=True,
                        help="path to the lst file containing the file paths of the page xml's to be processed")

    parser.add_argument('--des_dist', type=int, default=50,
                        help="desired distance (measured in pixels) of two adjacent pixels in the normed polygons")
    parser.add_argument('--max_d', type=int, default=100,
                        help="maximum distance (measured in pixels) for the calculation of the interline distances")
    parser.add_argument('--alpha', type=float, default=75,
                        help="alpha value for the alpha shape algorithm "
                             "(for alpha -> infinity we get the convex hulls, recommended: alpha >= des_dist)")
    parser.add_argument('--use_java_code', type=bool, default=True,
                        help="usage of methods written in java (faster than python!) or not")

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
