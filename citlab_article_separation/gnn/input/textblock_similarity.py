import logging
from random import seed
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors


def normalized_cos_sim(x, y):
    """
    cosine similarity, mapped linearly to [0,1] to look like confidence values

    Args:
        x (vector): first vector
        y (vector): second vector

    Returns:
        float: confidence-like value in [0,1]
    """
    cos = 0
    if np.any(x) and np.any(y):
        cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return (cos + 1) / 2


class TextblockSimilarity(object):
    """
    class for text block similarity computation

    Usage::

        featExt = TbSim(language=txtLanguage, wvFilePath=modelFilePath)
        featExt.minTbLength = 10
        featExt.defaultEdgeFeatureValue = [.5]
        …
        featExt.set_tb_dict(tbDict)
        featExt.run()
        print(featExt.featDict)
    """

    #:  :obj:list of float: default edge feature value, if edge not further considered
    defaultEdgeFeatureValue = [0.5]
    #:  int: minimum number of tokens in text block to be considered
    minTbLength = 5
    #:  dict: output dict containing feature values
    feature_dict = None

    def __init__(self, language, wv_path):
        """
        standard constructor

        Args:
            language: language name
            wv_path: word vector resource
        """
        self._language = language
        self._wordVectors = KeyedVectors.load(str(wv_path)).wv
        self._stopWords = stopwords.words(self._language)
        seed()
        self._tb_dict = None

    def set_tb_dict(self, tb_dict):
        """
        set dict with textblock data

        Args:
            tb_dict: dict with textblock data
        """
        self._tb_dict = tb_dict

    def run(self):
        """run computation"""
        self.feature_dict = {'edge_features': {'default': self.defaultEdgeFeatureValue}}
        self._calcBlockScores()
        self._calcEdgeScores()

    def _calcBlockScores(self):
        #   evaluating textblocks
        self._scoreDict = {}
        for (tbKey, tbData) in self._tb_dict.items():
            tokens = word_tokenize(text=tbData, language=self._language)
            tbSize = len(tokens)
            if tbSize < self.minTbLength:
                logging.debug(f"ignoring textblock {tbKey} with only {tbSize} words")
            else:
                logging.debug(f"processing textblock {tbKey} with {tbSize} words")
                words = [word for word in tokens if word.isalpha()]
                noStop = list(map(str.lower, [word for word in words if not word in self._stopWords]))
                vectList = [self._wordVectors.wv[word] for word in noStop if word in self._wordVectors]
                self._scoreDict[tbKey] = np.sum(vectList, axis=0, initial=0)
        logging.debug(f'evaluated {len(self._scoreDict)} textblocks …')

    def _calcEdgeScores(self):
        #   comparing textblocks
        cntPairs = 0
        tbKeys = sorted(self._scoreDict.keys())
        for tb0Key in tbKeys:
            self.feature_dict[tb0Key] = {}
            for tb1Key in tbKeys:
                if tb0Key < tb1Key:
                    cntPairs += 1
                    self.feature_dict[tb0Key][tb1Key] = [normalized_cos_sim(
                        self._scoreDict[tb0Key], self._scoreDict[tb1Key])]
                elif tb0Key > tb1Key:
                    self.feature_dict[tb0Key][tb1Key] = self.feature_dict[tb1Key][tb0Key]
        logging.debug(f'compared {cntPairs} textblock pairs …')


if __name__ == '__main__':
    lang = 'german'
    wv_path = '/home/johannes/devel/projects/tf_rel/workshop/roger/code/resWV/newseye_de_300.w2v'
    feature_extractor = TextblockSimilarity(language=lang, wv_path=wv_path)

    import json
    json_path = '/home/johannes/devel/projects/tf_rel/data/NewsEye_GT/onb_230_BC.json'
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    tb_dict = dict()
    for page_data in json_data['page']:
        page_name = page_data['page_file']
        tb_dict[page_name] = dict()
        for article_data in page_data['articles']:
            tb_id = article_data['text_blocks'][0]['text_block_id']
            tb_text = article_data['text_blocks'][0]['text']
            tb_dict[page_name][tb_id] = tb_text

    for page in tb_dict:
        print(page)
        feature_extractor.set_tb_dict(tb_dict[page])
        feature_extractor.run()
        for tb in feature_extractor.feature_dict:
            print(tb, "---", feature_extractor.feature_dict[tb])
        break

    ############################################################
    page_path = "/home/johannes/devel/data/NewsEye_GT/AS_BC/NewsEye_ONB_232_textblocks/274954/ONB_ibn_19110701_corrected_duplicated/page/ONB_ibn_19110701_009.xml"
    from citlab_python_util.parser.xml.page.page import Page
    page_file = Page(page_path)
    regions = page_file.get_regions()
    text_regions = regions['TextRegion']

    # build {tb : text} dict
    tb_dict = dict()
    for text_region in text_regions:
        # text = ""
        # for text_line in text_region.text_lines:
        #     text += text_line.text + "\n"
        text = "\n".join([text_line.text for text_line in text_region.text_lines])
        tb_dict[text_region.id] = text
    # run feature extractor
    feature_extractor.set_tb_dict(tb_dict)
    feature_extractor.run()
    for tb in feature_extractor.feature_dict:
        print(tb, "---", feature_extractor.feature_dict[tb])
