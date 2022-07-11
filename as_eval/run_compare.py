import argparse
import logging
import sys
import os
import glob
import as_eval.asQcTools as aqt


def find_dirs(name, root='.', exclude=None):
    results = []
    for path, dirs, files in os.walk(root):
        if name in dirs:
            # return os.path.join(path, name)
            results.append(os.path.join(path, name))
    if exclude:
        for ex in exclude.split(","):
            results = [res for res in results if ex not in res]
    return results


def setup_logger(name=None, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(levelname)7s - %(module)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_list", type=str, help="list file containing GT file paths", default=None)
    parser.add_argument("--gt_dir", type=str, help="dir path containing GT files", default=None)
    parser.add_argument("--exclude", type=str, help="comma separated strings to exclude from gt_dir", default=None)
    parser.add_argument("--work_dir", type=str, help="dir path containing clustering folders", required=True)
    parser.add_argument("--out_dir", type=str, help="dir path for the ouput files", required=True)
    parser.add_argument("--name", type=str, help="optional name for output comparison file", default=None)
    args = parser.parse_args()
    logger = setup_logger()

    # IN paths
    if args.gt_dir and args.gt_list:
        logger.error(f"Only one GT variant can be chosen at a time!")
        exit(1)
    elif args.gt_dir:
        gt_path = find_dirs("page", root=args.gt_dir)[0]
        gt_files = [os.path.join(gt_path, file_path) for file_path in glob.glob1(gt_path, '*.xml')]
    elif args.gt_list:
        gt_files = [path.rstrip() for path in open(args.gt_list, "r")]
    else:
        logger.error(f"Either --gt_list or --gt_dir is needed!")
        exit(1)
    clustering_paths = find_dirs("clustering", root=args.work_dir, exclude=args.exclude)
    logger.info(f"Using clustering paths:")
    for path in clustering_paths:
        logger.info(f"\t{path}")

    # OUT paths
    out_name = f"{args.name}_comparison" if args.name else "comparison"
    xlsx_out_path = os.path.join(args.out_dir, f"{out_name}.xlsx")
    # json_out_path = os.path.join(args.out_dir, "problems", "out.json")
    # stat_out_path = os.path.join(args.out_dir, "problems", "out.txt")
    # prob_out_path = os.path.join(args.out_dir, "problems", "out.xlsx")

    # #   initialize article separation checking
    # probSet = {
    #     aqt.AsProbCode.TL_11, aqt.AsProbCode.TL_12,
    #     aqt.AsProbCode.TR_11,
    #     aqt.AsProbCode.TL_21, aqt.AsProbCode.TL_22
    # }
    # asChecker = aqt.AsChecker(probSet)
    # asChecker.rootPath = workDirPath
    # asChecker.pageList = list(gtDirPath.glob('*.xml'))
    # # for methodDirPath in hypDirPath.iterdir():
    # #     asChecker.pageList.extend(list(methodDirPath.glob('*.xml')))
    # asChecker.checkPages()
    # logger.info(f'{asChecker.cntProbs} problems detected')
    # with jsonFilePath.open(mode='wt') as jsonFile:
    #     print(asChecker.probToJSON(), file=jsonFile)
    # with statFilePath.open(mode='wt') as statFile:
    #     print(asChecker.cntDict, file=statFile)
    #     print(f'\n{aqt.AsProbDesc().toString()}', file=statFile)
    # asChecker.probToXLSX(xlsxFilePath=probFilePath)
    # sys.exit()

    #   initialize comparison engine & result container
    asComper = aqt.SepPageBlComper()
    spcDict = aqt.SepPageCompDict()
    for gt_file_path in gt_files:
        logger.info(f'comparing GT from {gt_file_path} …')
        asComper.loadGT(gt_file_path)
        for clustering_path in clustering_paths:
            method_folders = [os.path.join(clustering_path, folder) for folder in os.listdir(clustering_path)]
            if args.exclude:
                for ex in args.exclude.split(","):
                    method_folders = [res for res in method_folders if ex not in res]
            for method_path in [folder for folder in method_folders if os.path.isdir(folder)]:
                cluster_file_path = os.path.splitext(os.path.basename(gt_file_path))[0] + "_clustering.xml"
                hyp_file_path = os.path.join(method_path, cluster_file_path)
                compRes = asComper.compareTo(hyp_file_path)
                logger.info(f'\t… with HYP in {hyp_file_path}:\t{compRes.__dict__}')
                spcDict.addItem(dataSet='Koeln111_test', gtXML=str(gt_file_path), hypXML=str(hyp_file_path),
                                spcDict=compRes)

    #   compare methods
    methEvaler = aqt.CompDictEvaler(spcDict=spcDict)
    methEvaler.calcWinnerDict()
    methEvaler.winnerStat2xlsx(xlsxFilePath=xlsx_out_path)
    logger.info(f'writing to {xlsx_out_path}')
