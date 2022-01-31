"""
Tools collection for checking article separations
"""

import logging
from pathlib import Path, PurePosixPath
from enum import Enum, auto, unique
from json import JSONEncoder
from typing import Any, Union, Set, List

from python_util.parser.xml.page.page import Page

logger = logging.getLogger(__name__)


@unique
class AsProbCode(Enum):
    """enum for problem codes"""
    TL_11 = auto()
    TL_12 = auto()
    TL_21 = auto()
    TR_11 = auto()


asProbCodeDesc = {
    AsProbCode.TL_11: "textline without text",
    AsProbCode.TL_12: "textline without article_id",
    AsProbCode.TL_21: "different textlines with identical text",
    AsProbCode.TR_11: "textregion with multiple article_ids",
}

asCheckerList = [
    ([AsProbCode.TL_11, AsProbCode.TL_12], 'checkTL1'),
    ([AsProbCode.TL_21], 'checkTL2'),
    ([AsProbCode.TR_11], 'checkTR'),
]


class AsProbDesc(dict):
    """container class for problem codes with description"""

    def __init__(self):
        super(AsProbDesc, self).__init__(asProbCodeDesc)

    def getWithDefault(self, code) -> str:
        return super().get(code, '(no description available)')

    def toString(self) -> str:
        """text representation of problem codes & description"""
        repData = map(lambda x: f"{x[0].name}\t{x[1]}", sorted(self.items(), key=lambda x: x[0].value))
        repStr = "Problem Codes\n\t"
        repStr += '\n\t'.join(repData)
        return repStr


class AsProbDict:
    """container class for article separation problems"""

    def __init__(self, code, entity, remark=''):
        self.code: AsProbCode = code
        self.entity: str = entity
        self.remark: str = remark

    def toString(self) -> str:
        """text representation of a problem"""
        repStr = f'{self.code.name}\t{self.entity}\t{self.remark}'
        return repStr


def asProbJSON(asProbObj: Union[AsProbCode, AsProbDict]) -> Union[str, dict]:
    if isinstance(asProbObj, AsProbCode):
        return asProbObj.name
    elif isinstance(asProbObj, AsProbDict):
        return asProbObj.__dict__
    else:
        raise TypeError(f'type not JSON serializable: {type(asProbObj)}')


class AsChecker:
    """class for checker engine"""

    probDesc: AsProbDesc = AsProbDesc()

    @classmethod
    def _buildWorkList(cls, codeSet: Set[AsProbCode]) -> List[tuple]:
        workList: List = []
        usedCodeSet = set()
        for (codeList, methodBase) in asCheckerList:
            actCodeSet = set(codeList).intersection(codeSet)
            if len(actCodeSet) > 0:
                workList.append((getattr(cls, f'_{methodBase}'), actCodeSet))
                usedCodeSet = usedCodeSet.union(actCodeSet)
        for code in codeSet.difference(usedCodeSet):
            logger.warning(
                f'{code.name} not (yet) implemented » ignoring problem: {cls.probDesc.getWithDefault(code)}')
        if len(workList) > 0:
            logger.debug(f'accomplishing checks: {workList}')
            return (workList, usedCodeSet)
        else:
            raise RuntimeError(f'no checks to be performed » exciting')

    def __init__(self, codeSet: Set[AsProbCode]):
        (workList, usedCodeSet) = self._buildWorkList(codeSet=codeSet)
        self.workList: List[tuple] = workList
        self.pageList: List[Path] = []
        self.probDict: dict = {}
        self.cntProbs: int = 0
        self.cntDict: dict = {code.name: 0 for code in usedCodeSet}
        self.actCodeSet: Set[AsProbCode] = set()
        self.actPage: Union[Page, None] = None

    def probToJSON(self) -> str:
        """json representation of the check results"""
        jsonEncoder = JSONEncoder(indent=2, default=asProbJSON)
        if len(self.probDict) > 0:
            return jsonEncoder.encode(self.probDict)
        else:
            return jsonEncoder.encode('(no problems detected)')

    def checkPages(self) -> None:
        """runs checking all pages in page list"""

        for pagePath in self.pageList:
            pageName = str(PurePosixPath(pagePath))
            self.actPage = Page(path_to_xml=pageName)
            logger.info(f'checking page {pageName}')
            self.probDict[pageName] = []
            for (method, codeSet) in self.workList:
                logger.debug(f'applying {method} for {codeSet}')
                self.actCodeSet = codeSet
                probList = method(self)
                if len(probList) > 0:
                    logger.debug(f'problems detected {probList}')
                    self.probDict[pageName].extend(probList)
                    self.cntProbs += len(probList)
            if len(self.probDict[pageName]) == 0:
                del self.probDict[pageName]

    def _checkTL1(self) -> List[AsProbDict]:
        """checks for problem: textlines without text or article_id"""
        probList = []
        for textLine in self.actPage.get_textlines(ignore_redundant_textlines=True):
            if AsProbCode.TL_11 in self.actCodeSet:
                logger.debug(f'checking: {self.probDesc.getWithDefault(AsProbCode.TL_11)}')
                if len(textLine.text) == 0:
                    prob = AsProbDict(code=probCode, entity=textLine.id, remark='empty')
                    probList.append(prob)
                    self.cntDict[probCode.name] += 1
            if AsProbCode.TL_12 in self.actCodeSet:
                logger.debug(f'checking: {self.probDesc.getWithDefault(AsProbCode.TL_12)}')
                if textLine.get_article_id() is None:
                    prob = AsProbDict(code=probCode, entity=textLine.id, remark='w/o article')
                    probList.append(prob)
                    self.cntDict[probCode.name] += 1
        return probList

    def _checkTL2(self) -> List[AsProbDict]:
        """checks for problem: textlines without text or article_id"""
        probList = []
        if AsProbCode.TL_21 in self.actCodeSet:
            logger.debug(f'checking: {self.probDesc.getWithDefault(AsProbCode.TL_21)}')
            textLineList = sorted(self.actPage.get_textlines(ignore_redundant_textlines=True), key=lambda x: x.id)
            for (idx, textLine1) in enumerate(textLineList):
                for textLine2 in textLineList[idx + 1:]:
                    if len(textLine1.text) > 0 and textLine1.text == textLine2.text:
                        prob = AsProbDict(code=probCode, entity=textLine1.id, remark=f'same as {textLine2.id}')
                        probList.append(prob)
                        self.cntDict[probCode.name] += 1
        return probList

    def _checkTR(self) -> List[AsProbDict]:
        """checks for problem: textregions with multiple article_ids"""
        probList = []
        for textRegion in self.actPage.get_text_regions():
            if AsProbCode.TR_11 in self.actCodeSet:
                logger.debug(f'checking: {self.probDesc.getWithDefault(AsProbCode.TR_11)}')
                artIdSet = set()
                for textLine in textRegion.text_lines:
                    if textLine.get_article_id() is not None:
                        artIdSet.add(textLine.get_article_id())
                if len(artIdSet) > 1:
                    prob = AsProbDict(code=probCode, entity=textRegion.id, remark=str(artIdSet))
                    probList.append(prob)
                    self.cntDict[probCode.name] += 1
        return probList


if __name__ == '__main__':
    # print(AsProbDesc().toString())
    asChecker = AsChecker({AsProbCode.TL_11, AsProbCode.TL_21, AsProbCode.TR_11})
    print(asChecker.workList)
    print(asChecker.usedCodeSet)
    cntDict = {code.name: 0 for code in asChecker.usedCodeSet}
    print(cntDict)
    # asChecker.probDict['somePage'] = [AsProbDict(AsProbCode.TL_11, 'ID-1', 'Remark-1'),
    #                                   AsProbDict(AsProbCode.TL_21, 'ID-2', 'Remark-2')]
    # print(asChecker.probDict)
    # print(asChecker.probToJSON())
