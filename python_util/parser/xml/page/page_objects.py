# -*- coding: utf-8 -*-

"""
This file contains python classes for the most important PAGE-XML instances, e.g. different kind of regions
(e.g. TextRegion), text lines (TextLine) or words (Word).
"""
import numpy as np
import logging
from lxml import etree

import python_util.parser.xml.page.page_constants as page_const
from python_util.geometry.polygon import Polygon
from python_util.parser.xml.page import page_util

from typing import List, Tuple


def polygon_to_points(polygon):
    """
    Convert a Polygon object ``poly`` to a Points object.

    :param polygon: Polygon object that is converted to a list of (x, y) coordinates.
    :type polygon: Polygon
    :return: Polygon as a list of (x,y) coordinates.
    :rtype: List[Tuple[int, int]]
    """
    x, y = polygon.x_points, polygon.y_points

    return Points(list(zip(x, y)))


def string_to_points(s):
    """
    Convert a PAGE-XML valid string to a list of (x,y) values. Valid means e.g. "0,0 1,2 3,4" for the points (0, 0),
    (1, 2) and (3,4).

    :param s: The points given as a string in the format as defined in PAGE-XML.
    :type s: str
    :return: List of points as (x,y) coordinates.
    :rtype: List[Tuple[int, int]]
    """
    l_s = s.split(' ')
    l_xy = list()
    for s_pair in l_s:  # s_pair = 'x,y'
        try:
            (sx, sy) = s_pair.split(',')
            l_xy.append((int(sx), int(sy)))
        except ValueError:
            print("Can't convert string '{}' to a point.".format(s_pair))
            exit(1)

    return l_xy


class Points:
    """
    This class defines the Point instance of the PAGE-XML format.
    """
    def __init__(self, points_list):
        if type(points_list[0][0]) == float:
            self.points_list = [(int(x), int(y)) for x,y in points_list]
        else:
            self.points_list = points_list

    def to_string(self):
        """Convert self.points_list to a PageXml valid format:
        'x1,y1 x2,y2 ... xN,yN'.

        :return: PageXml valid string format of coordinates.
        """
        s = ""
        for pt in self.points_list:
            if s:
                s += " "
            s += "%s,%s" % (pt[0], pt[1])
        return s

    def to_polygon(self):
        x, y = np.transpose(self.points_list)

        return Polygon(x.tolist(), y.tolist(), n_points=len(x))


class Region:
    """
    This class defines the Region instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None, node_string=None):
        if _id is None:
            raise page_util.PageXmlException("Every Region must have a unique id.")
        self.id = _id
        if points is None:
            raise page_util.PageXmlException("Every Region must have coordinates.")
        self.points = Points(points) if points is not None else None
        self.custom = custom
        self.node_string = node_string

    def set_points(self, points):
        """
        Sets the points attribute.

        :param points: List of (x, y) coordinates.
        :return:
        """
        self.points = Points(points)

    def to_page_xml_node(self):
        """
        Convert to a DOM node.

        :return: Corresponding PAGE-XML DOM node for this region.
        """
        region_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, self.node_string))
        region_nd.set('id', str(self.id))
        if self.custom:
            region_nd.set('custom', page_util.format_custom_attr(self.custom))

        coords_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCOORDS))
        coords_nd.set('points', self.points.to_string())
        region_nd.append(coords_nd)

        return region_nd

    def get_reading_order(self):
        """
        Get the reading order of this region as defined in the custom attribute.

        :return:
        """
        try:
            return self.custom["readingOrder"]["index"]
        except KeyError:
            # print("Reading order index missing.")
            return None

    def set_reading_order(self, reading_order):
        """
        If `reading_order` is not None write it to the custom attribute of the region. Otherwise remove the current
        reading order attribute from the custom dict.

        :param reading_order: The reading order of the region.
        :return:
        """
        if reading_order:
            try:
                self.custom["readingOrder"]["index"] = str(reading_order)
            except KeyError:
                self.custom["readingOrder"] = {}
                self.custom["readingOrder"]["index"] = str(reading_order)
        else:
            try:
                self.custom.pop("readingOrder")
            except KeyError:
                pass


class TextRegion(Region):
    """
    This class defines the TextRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None, text_lines=None,
                 region_type=page_const.TextRegionTypes.sPARAGRAPH):
        super().__init__(_id, custom, points, node_string=page_const.sTEXTREGION)
        if text_lines is None:
            text_lines = []
        self.text_lines = text_lines
        self.region_type = region_type

    def to_page_xml_node(self):
        """
        Convert to a DOM node. Also needs to convert all subsequent elements like TextLines and Words to DOM nodes.

        :return: Corresponding PAGE-XML DOM node for this text region.
        """
        region_nd = super().to_page_xml_node()
        region_nd.set('type', self.region_type)
        region_text = ""

        for text_line in self.text_lines:
            text_line_nd = text_line.to_page_xml_node()
            if text_line_nd is not None:
                region_nd.append(text_line_nd)
                if region_text:
                    region_text = '\n'.join([region_text, text_line.text])
                else:
                    region_text = text_line.text

        if region_text:
            text_equiv_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sTEXTEQUIV))
            unicode_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sUNICODE))
            unicode_nd.text = region_text
            text_equiv_nd.append(unicode_nd)
            region_nd.append(text_equiv_nd)

        return region_nd


class ImageRegion(Region):
    """
    This class defines the ImageRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sIMAGEREGION)


class LineDrawingRegion(Region):
    """
    This class defines the LineDrawingRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sLINEDRAWINGREGION)


class GraphicRegion(Region):
    """
    This class defines the GraphicRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sGRAPHICREGION)


class TableRegion(Region):
    """
    This class defines the TableRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sTABLEREGION)


class ChartRegion(Region):
    """
    This class defines the ChartRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sCHARTREGION)


class SeparatorRegion(Region):
    """
    This class defines the SeparatorRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sSEPARATORREGION)

    def get_orientation(self):
        try:
            return self.custom['structure']['orientation']
        except KeyError:
            return None


class MathsRegion(Region):
    """
    This class defines the MathsRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sMATHSREGION)


class ChemRegion(Region):
    """
    This class defines the ChemRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sCHEMREGION)


class MusicRegion(Region):
    """
    This class defines the MusicRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sMUSICREGION)


class AdvertRegion(Region):
    """
    This class defines the AdvertRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sADVERTREGION)


class NoiseRegion(Region):
    """
    This class defines the NoiseRegioni instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sNOISEREGION)


class UnknownRegion(Region):
    """
    This class defines the UnknownRegion instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, points=None):
        super().__init__(_id, custom, points, node_string=page_const.sUNKNOWNREGION)


class TextLine:
    """
    This class defines the TextLine instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, text=None, baseline=None, surr_p=None, words=None):
        if _id is None:
            raise page_util.PageXmlException("Every TextLine must have a unique id.")
        self.id = _id  # unique id of textline (str)
        # dictionary of dictionaries, e.g. {'readingOrder':{ 'index':'4' },'structure':{'type':'catch-word'}}
        self.custom = custom  # custom attr holding information like article id (dict of dicts)
        self.baseline = Points(baseline) if baseline is not None else None  # baseline of textline (Points object)
        self.text = text if text is not None else ""  # text present in the textline
        self.surr_p = Points(surr_p) if surr_p is not None else None  # surrounding polygon of textline (Points object)
        if words is None:
            words = []
        self.words = words

    def to_page_xml_node(self):
        """
        Convert to a DOM node. Also needs to convert all subsequent Word elements to DOM nodes if there are any.

        :return: Corresponding PAGE-XML DOM node for this text line.
        """
        text_line_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sTEXTLINE))
        text_line_nd.set('id', str(self.id))
        if self.custom:
            text_line_nd.set('custom', page_util.format_custom_attr(self.custom))

        if not self.surr_p:
            logging.warning(f"Can't convert TextLine to PAGE-XML node since no surrounding polygon is given ({self.id}).")
            return None

        coords_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCOORDS))
        coords_nd.set('points', self.surr_p.to_string())
        text_line_nd.append(coords_nd)

        if self.baseline:
            baseline_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sBASELINE))
            baseline_nd.set('points', self.baseline.to_string())
            text_line_nd.append(baseline_nd)

        for word in self.words:
            word_nd = word.to_page_xml_node()
            if word_nd is not None:
                text_line_nd.append(word_nd)

        if self.text is not None:
            text_equiv_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sTEXTEQUIV))
            unicode_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sUNICODE))
            unicode_nd.text = self.text
            text_equiv_nd.append(unicode_nd)
            text_line_nd.append(text_equiv_nd)

        return text_line_nd

    def set_points(self, points):
        """
        Sets the coordinates (``points``) for the text line (the surrounding polygon of the corresponding baseline).

        :param points: List of (x, y) coordinates that defines the text line.
        :return:
        """
        self.surr_p = Points(points)

    def set_baseline(self, baseline):
        """
        Set the coordinates for the corresponding baseline that is associated with this text line.

        :param baseline: A List of (x, y) coordinates defining the baseline.
        :return:
        """
        self.baseline = Points(baseline) if baseline is not None else None

    def get_reading_order(self):
        """
        Get the reading order of the this text line if there is any, otherwise return ``None``.

        :return: Reading order of the text lines if there is any, otherwise None.
        """
        try:
            return self.custom["readingOrder"]["index"]
        except KeyError:
            # print("Reading order index missing.")
            return None

    def get_article_id(self):
        """
        Get the article id of this text line.

        :return: Article ID.
        """
        try:
            return self.custom["structure"]["id"] if self.custom["structure"]["type"] == "article" else None
        except KeyError:
            return None

    def get_semantic_type(self):
        """
        Returns the semantic type of the text line, which is stored in the custom attribute (e.g. "heading").

        :return: The semantic type of the text line.
        """
        try:
            return self.custom["structure"]["semantic_type"]
        except KeyError:
            return None

    def set_reading_order(self, reading_order):
        """
        Sets the reading order of this text line if given, otherwise remove it from the custom dict.

        :param reading_order: The reading order of the text line.
        :return:
        """
        if reading_order:
            try:
                self.custom["readingOrder"]["index"] = str(reading_order)
            except KeyError:
                self.custom["readingOrder"] = {}
                self.custom["readingOrder"]["index"] = str(reading_order)
        else:
            try:
                self.custom.pop("readingOrder")
            except KeyError:
                pass

    def set_article_id(self, article_id=None):
        """
        Sets the article id (``article_id``) of this text line.
        :param article_id: The article id to set.
        :return:
        """
        if article_id:
            try:
                self.custom["structure"]["id"] = str(article_id)
            except KeyError:
                self.custom["structure"] = {}
                self.custom["structure"]["id"] = str(article_id)
            self.custom["structure"]["type"] = "article"
        else:
            try:
                self.custom['structure'].pop('id')
                if not self.custom['structure']:
                    self.custom.pop('structure')
            except KeyError:
                pass

    def set_structure_attribute(self, attribute_name, attribute):
        """
        Sets the value (``attribute``) for a a structure attribute (``attribute_name``), e.g. a value for
        "readingOrder".
        :param attribute_name: The name of the structure attribute.
        :param attribute: The value to set for the given attribute.
        :return:
        """
        try:
            self.custom["structure"][attribute_name] = str(attribute)
        except KeyError:
            self.custom["structure"] = {}
            self.custom["structure"][attribute_name] = str(attribute)


class Word:
    """
    This class defines the Word instance of the PAGE-XML format.
    """
    def __init__(self, _id, custom=None, text=None, surr_p=None):
        if _id is None:
            raise page_util.PageXmlException("Every Word must have a unique id.")
        self.id = _id  # unique id of textline (str)
        # dictionary of dictionaries, e.g. {'readingOrder':{ 'index':'4' },'structure':{'type':'catch-word'}}
        self.custom = custom  # custom attr holding information like reading order (dict of dicts)
        self.text = text if text is not None else ""  # text present in the textline
        self.surr_p = Points(surr_p) if surr_p is not None else None  # surrounding polygon of textline (Points object)

    def to_page_xml_node(self):
        """
        Convert to a DOM node.

        :return: Corresponding PAGE-XML DOM node for this text line.
        """
        word_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sWORD))
        word_nd.set('id', str(self.id))
        if self.custom:
            word_nd.set('custom', page_util.format_custom_attr(self.custom))

        if not self.surr_p:
            logging.warning(f"Can't convert Word to PAGE-XML node since no surrounding polygon is given ({self.id}).")
            return None

        coords_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCOORDS))
        coords_nd.set('points', self.surr_p.to_string())
        word_nd.append(coords_nd)

        if self.text is not None:
            text_equiv_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sTEXTEQUIV))
            unicode_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sUNICODE))
            unicode_nd.text = self.text
            text_equiv_nd.append(unicode_nd)
            word_nd.append(text_equiv_nd)

        return word_nd

    def set_points(self, points):
        """
        Sets the coordinates (``points``) for this word.
        :param points: List of (x, y) coordinates that defines the word.
        :return:
        """
        self.surr_p = Points(points)

    def get_reading_order(self):
        """
        Get the reading order value of this word.

        :return: The reading order value of this word.
        """
        try:
            return self.custom["readingOrder"]["index"]
        except KeyError:
            # print("Reading order index missing.")
            return None

    def set_reading_order(self, reading_order):
        """
        Set the reading order for this word if given, otherwise remove it from the custom dict.

        :param reading_order: The reading order value to set.
        :return:
        """
        if reading_order:
            try:
                self.custom["readingOrder"]["index"] = str(reading_order)
            except KeyError:
                self.custom["readingOrder"] = {}
                self.custom["readingOrder"]["index"] = str(reading_order)
        else:
            try:
                self.custom.pop("readingOrder")
            except KeyError:
                pass


# A dictionary of all region objects stored in a dictionary with the region name as their keys.
REGIONS_DICT = {page_const.sTEXTREGION: TextRegion, page_const.sIMAGEREGION: ImageRegion,
                page_const.sLINEDRAWINGREGION: LineDrawingRegion, page_const.sGRAPHICREGION: GraphicRegion,
                page_const.sTABLEREGION: TableRegion, page_const.sCHARTREGION: ChartRegion,
                page_const.sSEPARATORREGION: SeparatorRegion, page_const.sMATHSREGION: MathsRegion,
                page_const.sCHEMREGION: ChemRegion, page_const.sMUSICREGION: MusicRegion,
                page_const.sADVERTREGION: AdvertRegion, page_const.sNOISEREGION: NoiseRegion,
                page_const.sUNKNOWNREGION: UnknownRegion}

if __name__ == '__main__':
    points_polygon = [(1, 2), (3, 4), (5, 6)]
    surr_poly = [(0, 0), (0, 7), (7, 7), (7, 0)]
    points = Points(points_polygon)
    print(points.to_string())
    poly = points.to_polygon()
    print(poly.x_points, poly.y_points, poly.n_points)
    # points_copy = polygon_to_points(poly)
    # print(points_copy.to_string())

    text_line_1 = TextLine("tl1", custom={"readingOrder": {"index": 0}}, text="This is the first text line.",
                           baseline=points_polygon, surr_p=surr_poly)
    text_line_2 = TextLine("tl2", custom={"readingOrder": {"index": 1}}, text="This is the second text line.",
                           baseline=points_polygon, surr_p=surr_poly)
    print(text_line_1)
    text_line_nd = text_line_1.to_page_xml_node()
    print(etree.tostring(text_line_nd, pretty_print=True, encoding="UTF-8", standalone=True,
                         xml_declaration=True).decode("utf-8"))

    text_region = TextRegion("tr1", points=surr_poly, text_lines=[text_line_1, text_line_2])
    text_region_nd = text_region.to_page_xml_node()
    print(etree.tostring(text_region_nd, pretty_print=True).decode("utf-8"))
