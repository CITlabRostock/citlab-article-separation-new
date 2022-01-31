import logging
import os
import multiprocessing as mp
import python_util.basic.flags as flags
from python_util.basic.misc import split_list
from article_separation.gnn.input.feature_generation import generate_feature_jsons


flags.define_string('pagexml_list', '', 'input list with paths to pagexml files')
flags.define_string('out_dir', '', 'output directory for the json files')
flags.define_choices('interaction', ['fully', 'delaunay'], 'delaunay', str,
                     "('fully', 'delaunay')", 'determines the interacting nodes setup (graph edge layout)')
flags.define_boolean('visual_regions', False,
                     'Optionally build visual regions for nodes and edges (used for visual feature extraction)')
flags.define_choices('separators', ['line', 'bb'], 'bb', str, "('line', 'bb')",
                     'determines how edge separator features are generated. Either by intersection of the virtual '
                     'line connecting two text regions and separators, or by a rule-based bounding box approach.')
flags.define_list('external_jsons', str, 'STR', 'external json files containing additional features')
flags.define_string('wv_language', None, 'language used in tokenization nad stopwords filtering for text block '
                                         'similarities via word vectors')
flags.define_string('wv_path', None, 'path to wordvector embeddings used for text block similarities')
flags.define_integer('num_workers', 1, 'number of partitions to create from original list file and to compute in '
                                       'parallel. Only works when no external jsons are used.')
flags.FLAGS.parse_flags()
FLAGS = flags.FLAGS
if FLAGS.external_jsons:
    logging.info("Forced num_workers to 1, since external jsons are used.")
    FLAGS.num_workers = 1


if __name__ == '__main__':
    logging.getLogger().setLevel("INFO")
    flags.print_flags()

    # Get page paths
    page_paths = [os.path.abspath(line.rstrip()) for line in open(FLAGS.pagexml_list, "r")]
    n = FLAGS.num_workers

    # parallel over n workers (regarding the input list)
    if n > 1 and not FLAGS.external_jsons:
        processes = []
        for index, sublist in enumerate(split_list(page_paths, n)):
            # start worker
            p = mp.Process(target=generate_feature_jsons,
                           args=(sublist,
                                 FLAGS.out_dir,
                                 FLAGS.interaction,
                                 FLAGS.visual_regions,
                                 FLAGS.external_jsons,
                                 (FLAGS.wv_language, FLAGS.wv_path),
                                 FLAGS.separators))
            p.start()
            logging.info(f"Started worker {index}")
            processes.append(p)
        for p in processes:
            p.join()
        logging.info("All workers done.")
    # single threaded (forced if external jsons are used)
    else:
        generate_feature_jsons(page_paths,
                               FLAGS.out_dir,
                               FLAGS.interaction,
                               FLAGS.visual_regions,
                               FLAGS.external_jsons,
                               (FLAGS.wv_language, FLAGS.wv_path),
                               FLAGS.separators)
