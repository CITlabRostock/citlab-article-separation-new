"""minimal example run script"""

import logging
import sys
from pathlib import Path

import asQcTools as aqt

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='\t%(levelname)s\t%(module)s\n%(message)s')

    #   set up paths
    workDirPath = Path("../work")
    gtDirPath = workDirPath / "page"
    hypDirPath = workDirPath / "clustering"
    xlsxFilePath = (workDirPath / "minRunEx").with_suffix('.xlsx')
    jsonFilePath = (workDirPath / "minRunEx").with_suffix('.json')
    statFilePath = (workDirPath / "minRunEx").with_suffix('.txt')

    # #   initialize article separation checking
    # probSet = {aqt.AsProbCode.TL_11, aqt.AsProbCode.TL_12, aqt.AsProbCode.TR_11, aqt.AsProbCode.TL_21}
    # asChecker = aqt.AsChecker(probSet)
    # asChecker.pageList = list(gtDirPath.glob('*.xml'))
    # # for methodDirPath in hypDirPath.iterdir():
    # #     asChecker.pageList.extend(list(methodDirPath.glob('*.xml')))
    # asChecker.checkPages()
    # with jsonFilePath.open(mode='wt') as jsonFile:
    #     print(asChecker.probToJSON(), file=jsonFile)
    # with statFilePath.open(mode='wt') as statFile:
    #     print(asChecker.cntDict, file=statFile)
    #     print(f'\n{aqt.AsProbDesc().toString()}', file=statFile)
    # logging.info(f'{asChecker.cntProbs} problems detected')
    # # sys.exit()

    #   initialize comparison engine & result container
    asComper = aqt.SepPageBlComper()
    spcDict = aqt.SepPageCompDict()
    for gtFilePath in gtDirPath.glob('*.xml'):
        logging.info(f'comparing GT from {gtFilePath} …')
        asComper.loadGT(gtFilePath)
        for methodDirPath in hypDirPath.iterdir():
            hypFilePath = methodDirPath / gtFilePath.name
            compRes = asComper.compareTo(hypFilePath)
            logging.info(f'\t… with HYP in {hypFilePath}:\t{compRes.__dict__}')
            spcDict.addItem(dataSet='example', gtXML=str(gtFilePath), hypXML=str(hypFilePath),
                            spcDict=compRes)

    #   compare methods
    methEvaler = aqt.CompDictEvaler(spcDict=spcDict)
    methEvaler.calcWinnerDict()
    methEvaler.winnerStat2xlsx(xlsxFilePath=xlsxFilePath)
    logging.info(f'writing to {xlsxFilePath}')
