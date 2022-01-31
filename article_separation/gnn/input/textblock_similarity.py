import logging
import numpy as np
from random import seed
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors


def normalized_cos_sim(x, y):
    """
    Cosine similarity, mapped linearly to [0,1] to look like confidence values.

    :param x: first vector
    :param y: second vector
    :return: confidence-like value in [0,1]
    """
    cos = 0
    if np.any(x) and np.any(y):
        cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return (cos + 1) / 2


class TextblockSimilarity(object):
    """
    Class for text block similarity computation using language-specific word vectors.

    Usage::

        feat_extractor = TextblockSimilarity(language=text_language, wv_path=model_file_path)
        feat_extractor.min_tb_len = 5
        feat_extractor.default_edge_value = [.5]
        …
        feat_extractor.set_tb_dict(tb_dict)
        feat_extractor.run()
        print(feat_extractor.feature_dict)
    """

    #:  :obj:list of float: default edge feature value, if edge not further considered
    default_edge_value = [0.5]
    #:  int: minimum number of tokens in text block to be considered
    min_tb_len = 5
    #:  dict: output dict containing feature values
    feature_dict = None

    def __init__(self, language, wv_path):
        self._language = language
        self._wordVectors = KeyedVectors.load(str(wv_path)).wv
        self._stopWords = stopwords.words(self._language)
        seed()
        self._tb_dict = None

    def set_tb_dict(self, tb_dict):
        """
        Set dictionary with textblock data.

        :param tb_dict: dict with textblock data
        :return: None
        """
        self._tb_dict = tb_dict

    def run(self):
        """Run computation"""
        self.feature_dict = {'edge_features': {'default': self.default_edge_value}}
        self._calc_block_scores()
        self._calc_edge_scores()

    def _calc_block_scores(self):
        #   evaluating textblocks
        self._scoreDict = {}
        for (tbKey, tbData) in self._tb_dict.items():
            tokens = word_tokenize(text=tbData, language=self._language)
            tb_size = len(tokens)
            if tb_size < self.min_tb_len:
                logging.debug(f"ignoring textblock {tbKey} with only {tb_size} words")
            else:
                logging.debug(f"processing textblock {tbKey} with {tb_size} words")
                words = [word for word in tokens if word.isalpha()]
                no_stop = list(map(str.lower, [word for word in words if word not in self._stopWords]))
                vect_list = [self._wordVectors.wv[word] for word in no_stop if word in self._wordVectors]
                self._scoreDict[tbKey] = np.sum(vect_list, axis=0, initial=0)
        logging.debug(f'evaluated {len(self._scoreDict)} textblocks …')

    def _calc_edge_scores(self):
        #   comparing textblocks
        count_pairs = 0
        tb_keys = sorted(self._scoreDict.keys())
        for tb0Key in tb_keys:
            self.feature_dict['edge_features'][tb0Key] = {}
            for tb1Key in tb_keys:
                if tb0Key < tb1Key:
                    count_pairs += 1
                    self.feature_dict['edge_features'][tb0Key][tb1Key] = [normalized_cos_sim(
                        self._scoreDict[tb0Key], self._scoreDict[tb1Key])]
                elif tb0Key > tb1Key:
                    self.feature_dict['edge_features'][tb0Key][tb1Key] = \
                        self.feature_dict['edge_features'][tb1Key][tb0Key]
        logging.debug(f'compared {count_pairs} textblock pairs …')
