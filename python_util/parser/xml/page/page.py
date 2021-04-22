# -*- coding: utf-8 -*-
import datetime
import logging
import os
from argparse import ArgumentParser

import cssutils
from lxml import etree

import citlab_python_util.parser.xml.page.page_constants as page_const
from citlab_python_util.parser.xml.page import page_util, page_objects
from citlab_python_util.parser.xml.page.page_objects import TextLine, TextRegion, REGIONS_DICT, Word

# Make sure that the css parser for the custom attribute doesn't spam "WARNING Property: Unknown Property name."
cssutils.log.setLevel(logging.ERROR)

# logger.basicConfig(filename="docs/Page.log",
#                     format="%(asctime)s:%(levelname)s:%(message)s", filemode="w")  # add filemode="w" to overwrite file
logger = logging.getLogger("Page")


class Page:
    """
    Various utilities to deal with PageXml format
    """

    def __init__(self, path_to_xml=None, creator_name=page_const.sCREATOR, img_filename=None, img_w=None, img_h=None):
        self.page_doc = self.load_page_xml(path_to_xml) if path_to_xml is not None else self.create_page_xml_document(
            creator_name, img_filename, img_w, img_h)
        if len(self.page_doc.getroot().getchildren()) != 2:
            elts = self.page_doc.getroot().getchildren()
            # if Metadata node is missing, add it
            if page_const.sMETADATA_ELT not in [elt.tag for elt in elts]:
                self.create_metadata(page_const.sCREATOR, comments="Metadata entry was missing, added..")

        if not self.validate(self.page_doc):
            logger.debug("File given by {} is not a valid PageXml file.".format(path_to_xml))
            # exit(1)
        self.metadata = self.get_metadata()
        self.textlines = self.get_textlines()

    # =========== SCHEMA ===========

    def validate(self, doc):
        """
        Validate against the PageXml schema used by Transkribus

        Return True or False
        """
        if not page_const.cachedValidationContext:
            schema_filename_ = self.get_schema_filename()
            xmlschema_doc = etree.parse(schema_filename_)
            page_const.cachedValidationContext = etree.XMLSchema(xmlschema_doc)

        b_valid = page_const.cachedValidationContext.validate(doc)
        log = page_const.cachedValidationContext.error_log

        if not b_valid:
            logger.debug(log)
        return b_valid

    @classmethod
    def get_schema_filename(cls):
        """
        Return the path to the schema, built from the path of this module.
        """
        filename = os.path.join(os.path.dirname(__file__), page_const.XSL_SCHEMA_FILENAME)
        return filename

    # =========== METADATA ===========
    """
    <complexType name="MetadataType">
        <sequence>
            <element name="Creator" type="string"></element>
            <element name="Created" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="LastChange" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="Comments" type="string" minOccurs="0"
                maxOccurs="1"></element>
        </sequence>
    </complexType>
    """

    def get_metadata(self):
        """
        Parse the metadata of the PageXml DOM or of the given Metadata node
        return a Metadata object
        """
        _, nd_creator, nd_created, nd_last_change, nd_comments, nd_transkribus_meta = self._get_metadata_nodes()

        if nd_comments is not None and etree.QName(nd_comments.tag).localname == page_const.sTranskribusMetadata_ELT:
            nd_transkribus_meta = nd_comments
            nd_comments = None

        transkribus_meta = None
        if nd_transkribus_meta is not None:
            transkribus_meta = self._get_transkribus_meta_from_nd(nd_transkribus_meta)

        return Metadata(nd_creator.text
                        , nd_created.text
                        , nd_last_change.text
                        , nd_comments.text if nd_comments is not None else None
                        , transkribus_meta)

    def set_metadata(self, creator, comments=None):
        """
        Pass EITHER a DOM or a Metadata DOM node!! (and pass None for the other)
        Set the metadata of the PageXml DOM or of the given Metadata node

        Update the Created and LastChange fields.
        Either update the comments fields or delete it.

        You MUST indicate the creator (a string)
        You MAY give a comments (a string)
        The Created field is kept unchanged
        The LastChange field is automatically set.
        The comments field is either updated or deleted.
        return the Metadata DOM node
        """
        nd_metadata, nd_creator, nd_created, nd_last_change, nd_comments, nd_transkribus = self._get_metadata_nodes()
        # TODO: CHANGE nd_creator.text to something expressive/meaningful
        # if nd_creator.text and nd_creator.text != creator and ", modified by " + creator not in nd_creator.text:
        #     nd_creator.text += ", modified by " + creator
        # The schema seems to call for GMT date&time  (IMU)
        # ISO 8601 says:  "If the time is in UTC, add a Z directly after the time without a space. Z is the zone
        # designator for the zero UTC offset."
        # Python seems to break the standard unless one specifies properly a timezone by sub-classing tzinfo.
        # But too complex stuff so, I simply add a 'Z'
        nd_last_change.text = datetime.datetime.utcnow().isoformat() + "Z"
        if comments is not None:
            if nd_comments is None:  # we need to add one!
                nd_comments = etree.SubElement(nd_metadata, page_const.sCOMMENTS_ELT)
            nd_comments.text = comments
        return nd_metadata

    def create_metadata(self, creator_name=page_const.sCREATOR, comments=None):
        xml_page_root = self.page_doc.getroot()

        metadata = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sMETADATA_ELT))
        xml_page_root.insert(0, metadata)
        creator = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCREATOR_ELT))
        creator.text = creator_name
        created = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCREATED_ELT))
        created.text = datetime.datetime.utcnow().isoformat() + "Z"
        last_change = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sLAST_CHANGE_ELT))
        last_change.text = datetime.datetime.utcnow().isoformat() + "Z"
        comments_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCOMMENTS_ELT))
        comments_nd.text = comments

        metadata.append(creator)
        metadata.append(created)
        metadata.append(last_change)
        metadata.append(comments_nd)

        return metadata

    def _get_metadata_nodes(self):
        """
        Parse the metadata of the PageXml DOM or of the given Metadata node
        return a 4-tuple:
            DOM nodes of Metadata, Creator, Created, Last_Change, Comments (or None if no comments)
        """
        l_nd = self.get_child_by_name(self.page_doc, page_const.sMETADATA_ELT)
        if len(l_nd) != 1:
            raise ValueError(
                "PageXml should have exactly one %s node but found %s" % (page_const.sMETADATA_ELT, str(len(l_nd))))
        dom_nd = l_nd[0]
        assert etree.QName(dom_nd.tag).localname == page_const.sMETADATA_ELT
        nd1 = dom_nd[0]

        if etree.QName(nd1.tag).localname != page_const.sCREATOR_ELT:
            raise ValueError("PageXMl mal-formed Metadata: Creator element must be 1st element")

        nd2 = nd1.getnext()
        if etree.QName(nd2.tag).localname != page_const.sCREATED_ELT:
            raise ValueError("PageXMl mal-formed Metadata: Created element must be 2nd element")

        nd3 = nd2.getnext()
        if etree.QName(nd3.tag).localname != page_const.sLAST_CHANGE_ELT:
            raise ValueError("PageXMl mal-formed Metadata: LastChange element must be 3rd element")

        nd4 = nd3.getnext()
        if nd4 is not None:
            if etree.QName(nd4.tag).localname not in [page_const.sCOMMENTS_ELT, page_const.sTranskribusMetadata_ELT]:
                raise ValueError("PageXMl mal-formed Metadata: Comments or TranskribusMetadata element must be 4th element")

        nd5 = nd4.getnext() if nd4 is not None else None
        if nd5 is not None:
            if etree.QName(nd5.tag).localname != page_const.sTranskribusMetadata_ELT:
                raise ValueError("PageXMl mal-formed Metadata: TranskribusMetadata element must be the 5th element")

        return dom_nd, nd1, nd2, nd3, nd4, nd5

    # =========== XML STUFF ===========
    @classmethod
    def get_child_by_name(cls, elt, s_child_name):
        """
        look for all child elements having that name in PageXml namespace!!!
            Example: lNd = PageXMl.get_child_by_name(elt, "Baseline")
        return a DOM node
        """
        # return elt.findall(".//{%s}:%s"%(cls.NS_PAGE_XML,s_child_name))
        return elt.xpath(".//pc:%s" % s_child_name, namespaces={"pc": page_const.NS_PAGE_XML})

    def get_ancestor_by_name(self, elt, s_name):
        return elt.xpath("ancestor::pc:%s" % s_name, namespaces={"pc": page_const.NS_PAGE_XML})

    @classmethod
    def get_child_by_id(cls, elt, _id):
        """
        look for all child elements having that id
            Example: lNd = PageXMl.get_child_by_id(elt, "tl_2")
        return a DOM node
        """
        return elt.xpath(".//*[@id='%s']" % _id)

    @classmethod
    def get_ancestor_by_id(cls, elt, _id):
        """
        look for all ancestor elements having that id
            Example: lNd = PageXMl.get_ancestor_by_name(elt, "tl_2")
        return a DOM node
        """
        return elt.xpath("ancestor::*[@id='%s']" % _id)

    def get_custom_attr(self, nd, s_attr_name, s_sub_attr_name=None):
        """
        Read the custom attribute, parse it, and extract the 1st or 1st and 2nd key value
        e.g. get_custom_attr(nd, "structure", "type")     -->  "catch-word"
        e.g. get_custom_attr(nd, "structure")             -->  {'type':'catch-word', "toto", "tutu"}
        return a dictionary if no 2nd key provided, or a string if 1st and 2nd key provided
        Raise KeyError if one of the attribute does not exist
        """
        c_node = nd.get(page_const.sCUSTOM_ATTR)
        if c_node is None:
            return None
        ddic = self.parse_custom_attr(c_node)

        # First key

    def set_custom_attr_from_dict(self, nd, custom_dict):
        nd.set(page_const.sCUSTOM_ATTR, page_util.format_custom_attr(custom_dict))
        return nd

    def set_custom_attr(self, nd, s_attr_name, s_sub_attr_name, s_val):
        """
        Change the custom attribute by setting the value of the 1st+2nd key in the DOM
        return the value
        Raise KeyError if one of the attributes does not exist
        """
        ddic = self.parse_custom_attr(nd.get(page_const.sCUSTOM_ATTR))
        try:
            ddic[s_attr_name][s_sub_attr_name] = str(s_val)
        except KeyError:
            ddic[s_attr_name] = dict()
            ddic[s_attr_name][s_sub_attr_name] = str(s_val)

        sddic = page_util.format_custom_attr(ddic)
        nd.set(page_const.sCUSTOM_ATTR, sddic)
        return s_val

    def remove_custom_attr(self, nd, s_attr_name, s_sub_attr_name):
        ddic = self.parse_custom_attr(nd.get(page_const.sCUSTOM_ATTR))
        if s_attr_name in ddic and s_sub_attr_name in ddic[s_attr_name]:
            ddic[s_attr_name].pop(s_sub_attr_name)
        else:
            print("Can't remove {} from {} in {}.".format(s_sub_attr_name, s_attr_name, ddic))

    @staticmethod
    def parse_custom_attr(s):
        """
        The custom attribute contains data in a CSS style syntax.
        We parse this syntax here and return a dictionary of dictionaries

        Example:
        parse_custom_attr( "readingOrder {index:4;} structure {type:catch-word;}" )
            --> { 'readingOrder': { 'index':'4' }, 'structure':{'type':'catch-word'} }
        """
        if not s:
            return {}
        custom_dict = {}
        sheet = cssutils.parseString(s)
        for rule in sheet:
            selector = rule.selectorText
            prop_dict = {}
            for prop in rule.style:
                prop_dict[prop.name] = prop.value
            custom_dict[selector] = prop_dict

        return custom_dict

    @classmethod
    def get_text_equiv(cls, nd):
        textequiv = cls.get_child_by_name(nd, page_const.sTEXTEQUIV)
        if not textequiv:
            return ''
        # TODO: Maybe replace by getting the first entry of just one hierarchy below,
        #  e.g.for TextLine ignoring the Word data
        text = cls.get_child_by_name(textequiv[-1], "Unicode")
        if not text:
            return ''
        return text[0].text

    @staticmethod
    def make_text(nd):
        """
        build the text of a sub-tree by considering that textual nodes are tokens to be concatenated, with a space as separator
        NO! (JLM 2018)return None if no textual node found

        return empty string if no text node found
        """
        return " ".join(nd.itertext())

    # =========== GEOMETRY ===========
    @staticmethod
    def get_point_list(data):
        """
        get either an XML node of a PageXml object
              , or the content of a points attribute, e.g.
                1340,240 1696,240 1696,304 1340,304
        return the list of (x,y) of the polygon of the object - ( it is a list of int tuples)
        """
        try:
            ls_pair = data.split(' ')
        except AttributeError:
            lnd_points = data.xpath("(.//@points)[1]")
            s_points = lnd_points[0]
            ls_pair = s_points.split(' ')
        try:
            l_xy = list()
            for s_pair in ls_pair:  # s_pair = 'x,y'
                (sx, sy) = s_pair.split(',')
                l_xy.append((int(sx), int(sy)))
        except ValueError:
            return None
        return l_xy

    @staticmethod
    def set_points(nd, l_xy):
        """
        set the points attribute of that node to reflect the l_xy values
        if nd is None, only returns the string that should be set to the @points attribute
        return the content of the @points attribute
        """
        s_pairs = " ".join(["%d,%d" % (int(x), int(y)) for x, y in l_xy])
        if nd is not None:
            nd.set("points", s_pairs)
        return s_pairs

    # ======== ARTICLE STUFF =========

    def get_article_dict(self):
        article_dict = {}
        for tl in self.textlines:
            a_id = tl.get_article_id()
            if a_id in article_dict:
                article_dict[a_id].append(tl)
            else:
                article_dict[a_id] = [tl]

        return article_dict

    def get_image_resolution(self):
        page_nd = self.get_child_by_name(self.page_doc, "Page")[0]
        img_width = int(page_nd.get("imageWidth"))
        img_height = int(page_nd.get("imageHeight"))

        return img_width, img_height

    def get_print_space_coords(self):
        ps_nd = self.get_child_by_name(self.page_doc, page_const.sPRINT_SPACE)

        if len(ps_nd) != 1:
            print(f"Expected exactly one {page_const.sPRINT_SPACE} node, but got {len(ps_nd)}.")
            # exit(1)
            print(f"Fallback to image size.")
            img_width, img_height = self.get_image_resolution()

            ps_coords = [(0, 0), (img_width, 0), (img_width, img_height), (0, img_height)]

        else:
            ps_nd = ps_nd[0]

            # we assume that the PrintSpace is given as a rectangle, thus having four coordinates
            ps_coords = self.get_point_list(
                self.get_child_by_name(ps_nd, page_const.sCOORDS)[0].get(page_const.sPOINTS_ATTR))
            for i, (x, y) in enumerate(ps_coords):
                if x < 0:
                    x_new = 0
                else:
                    x_new = x
                if y < 0:
                    y_new = 0
                else:
                    y_new = y
                ps_coords[i] = (x_new, y_new)

            if len(ps_coords) != 4:
                print(f"Expected exactly four rectangle coordinates, but got {len(ps_coords)}.")
                exit(1)

        return ps_coords

    def get_ids(self):
        """
        Return a list of all current ids used in the PAGE object.
        :return: list of ids
        """
        return self.page_doc.xpath("//@id")

    def get_unique_id(self, page_object_name):
        """
        For a specific page object with name `page_object_name` (e.g. TextRegion or TextLine) find an ID that is not
        already taken. The new ID has the format `{page_object_name}_[1-9][0-9]*`
        :param page_object_name: type of the page object, e.g. TextRegion or TextLine
        :return: a unique ID for the Page object
        """
        existing_ids = self.get_ids()
        for i in range(1000):
            new_id = page_object_name + "_" + str(i + 1)
            if new_id not in existing_ids:
                return new_id
        return None

    def get_text_regions(self, text_region_type=None):
        text_region_nds = self.get_child_by_name(self.page_doc, page_const.sTEXTREGION)
        res = []
        if len(text_region_nds) > 0:
            for text_region in text_region_nds:
                text_region_nd_type = text_region.get('type')
                tr_type = text_region_nd_type if text_region_nd_type is not None else page_const.TextRegionTypes.sPARAGRAPH
                if text_region_type is not None and tr_type != text_region_type:
                    continue
                text_region_id = text_region.get("id")
                text_region_custom_attr = self.parse_custom_attr(text_region.get(page_const.sCUSTOM_ATTR))
                text_region_coords = self.get_point_list(
                    self.get_child_by_name(text_region, page_const.sCOORDS)[0].get(page_const.sPOINTS_ATTR))
                text_region_text_lines = self.get_textlines(text_region)

                tr = TextRegion(text_region_id, text_region_custom_attr, text_region_coords, text_region_text_lines,
                                tr_type)
                res.append(tr)

        return res

    def remove_regions(self, region_type):
        if region_type not in REGIONS_DICT:
            logger.info("There is no region with type {}, skipping.".format(region_type))
            return

        r_nds = self.get_child_by_name(self.page_doc, region_type)
        if len(r_nds) == 0:
            logger.info("No regions with of type {} found in this PAGE file.".format(region_type))
            return

        logger.info("Removing all regions of type {}.".format(region_type))
        for r_nd in r_nds:
            self.remove_page_xml_node(r_nd)

    def get_regions(self):
        res = {}
        for r_name in REGIONS_DICT.keys():
            if r_name == page_const.sTEXTREGION:
                text_regions = self.get_text_regions()
                if len(text_regions) > 0:
                    res[r_name] = text_regions
                continue
            r_nds = self.get_child_by_name(self.page_doc, r_name)
            if len(r_nds) > 0:
                r_class = REGIONS_DICT[r_name]
                res[r_name] = [r_class(reg.get("id"), self.parse_custom_attr(reg.get(page_const.sCUSTOM_ATTR)),
                                       self.get_point_list(
                                           self.get_child_by_name(reg, page_const.sCOORDS)[0].get(
                                               page_const.sPOINTS_ATTR)))
                               for reg in r_nds]
        return res

    def get_textlines(self, text_region_nd=None, ignore_redundant_textlines=True):
        if text_region_nd is not None:
            tl_nds = self.get_child_by_name(text_region_nd, page_const.sTEXTLINE)
        else:
            tl_nds = self.get_child_by_name(self.page_doc, page_const.sTEXTLINE)

        res = []
        tl_id_set = set()
        for tl in tl_nds:
            tl_id = tl.get("id")
            if tl_id in tl_id_set and ignore_redundant_textlines:
                continue
            tl_id_set.add(tl_id)
            tl_custom_attr = self.parse_custom_attr(tl.get(page_const.sCUSTOM_ATTR))
            tl_text = self.get_text_equiv(tl)
            tl_bl_nd = self.get_child_by_name(tl, page_const.sBASELINE)
            tl_bl = self.get_point_list(tl_bl_nd[0]) if tl_bl_nd else None
            tl_surr_p = self.get_point_list(tl)
            words = self.get_words(tl)
            res.append(TextLine(tl_id, tl_custom_attr, tl_text, tl_bl, tl_surr_p, words))


        # return [TextLine(tl.get("id"), self.parse_custom_attr(tl.get(self.sCUSTOM_ATTR)), self.get_text_equiv(tl),
        #                  self.get_point_list(self.get_child_by_name(tl, self.sBASELINE)[0]), self.get_point_list(tl))
        #         for tl in tl_nds]

        return res

    def get_words(self, text_line_nd=None, ignore_redundant_words=True):
        if text_line_nd is not None:
            word_nds = self.get_child_by_name(text_line_nd, page_const.sWORD)
        else:
            word_nds = self.get_child_by_name(self.page_doc, page_const.sWORD)

        res = []
        word_id_set = set()
        for word in word_nds:
            word_id = word.get("id")
            if word_id in word_id_set and ignore_redundant_words:
                continue
            word_id_set.add(word_id)
            word_custom_attr = self.parse_custom_attr(word.get(page_const.sCUSTOM_ATTR))
            word_text = self.get_text_equiv(word)
            word_surr_p = self.get_point_list(word)
            res.append(Word(word_id, word_custom_attr, word_text, word_surr_p))

        return res


    def update_textlines(self):
        self.textlines = self.get_textlines()

    def set_textline_attr(self, textlines):
        """

        :param textlines: list of TextLine objects
        :type textlines: list of TextLine
        :return: None
        """
        for tl in textlines:
            tl_nd = self.get_child_by_id(self.page_doc, tl.id)[0]
            self.set_custom_attr_from_dict(tl_nd, tl.custom)
            # for k, d in tl.custom.items():
            #     for k1, v1 in d.items():
            #         if v1 is None:
            #             self.remove_custom_attr(tl_nd, k, k1)
            #             break
            #         else:
            #             self.set_custom_attr(tl_nd, k, k1, v1)

            # if tl.get_article_id() is None:
            #     continue
            # tl_nd = cls.get_child_by_id(nd, tl.id)[0]
            # cls.set_custom_attr(tl_nd, "structure", "id", tl.get_article_id())
            # cls.set_custom_attr(tl_nd, "structure", "type", "article")

    def add_region(self, region, overwrite=False):
        # TODO: Check if region is overlapping with other regions. Add reading order.
        page_nd = self.get_child_by_name(self.page_doc, "Page")[0]

        region_id = region.id
        existent_region_nds = self.get_child_by_id(page_nd, region_id)

        region_nd = None
        if len(existent_region_nds) > 0:
            if overwrite:
                logger.debug("Region with id {} already existent, overwriting.".format(region_id))
                for existent_region_nd in existent_region_nds:
                    self.remove_page_xml_node(existent_region_nd)
                region_nd = region.to_page_xml_node()
            else:
                logger.debug("Region with id {} already existent, skipping.".format(region_id))
        else:
            region_nd = region.to_page_xml_node()
        if region_nd is not None:
            page_nd.append(region_nd)

    def set_text_regions(self, text_regions, overwrite=False):
        # TODO: Define behaviour for overwrite=False
        if overwrite:
            current_text_region_nds = self.get_child_by_name(self.page_doc, page_const.sTEXTREGION)
            for text_region_nd in current_text_region_nds:
                self.remove_page_xml_node(text_region_nd)

        page_nd = self.get_child_by_name(self.page_doc, "Page")[0]
        for text_region in text_regions:
            text_region_nd = text_region.to_page_xml_node()
            page_nd.append(text_region_nd)

    def set_text_lines(self, text_region, text_lines, overwrite=False):
        if type(text_region) == page_objects.TextRegion:
            text_region_nd = self.get_child_by_id(self.page_doc, text_region.id)[0]
        else:
            text_region_nd = text_region

        if overwrite:
            current_text_line_nds = self.get_child_by_name(text_region_nd, page_const.sTEXTLINE)
            for text_line_nd in current_text_line_nds:
                self.remove_page_xml_node(text_line_nd)

        new_text = ""
        last_text_line_nd = self.get_child_by_name(text_region_nd, page_const.sTEXTLINE)
        if last_text_line_nd:
            last_text_line_nd = last_text_line_nd[0]
            idx = text_region_nd.index(last_text_line_nd)
        else:
            idx = 0
        for text_line in text_lines:
            text_line_nd = text_line.to_page_xml_node()
            new_text = "\n".join([new_text, text_line.text])
            text_region_nd.insert(idx, text_line_nd)
            idx += 1

        unicode_nd = self.get_child_by_name(text_region_nd, page_const.sUNICODE)
        if unicode_nd:
            unicode_nd = unicode_nd[-1]
            unicode_nd.text = new_text
        else:
            unicode_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sUNICODE))
            unicode_nd.text = new_text

            text_equiv_nd = self.get_child_by_name(text_region_nd, page_const.sTEXTEQUIV)
            if text_equiv_nd:
                text_equiv_nd = text_equiv_nd[0]
                text_equiv_nd.append(unicode_nd)
                text_region_nd.append(text_equiv_nd)
            else:
                text_equiv_nd = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sTEXTEQUIV))
                text_equiv_nd.append(unicode_nd)

    # =========== CREATION ===========
    def create_page_xml_document(self, creator_name=page_const.sCREATOR, filename=None, img_w=0, img_h=0):
        """
            create a new PageXml document
        """
        xml_page_root = etree.Element('{%s}PcGts' % page_const.NS_PAGE_XML,
                                      attrib={"{" + page_const.NS_XSI + "}schemaLocation": page_const.XSILOCATION},
                                      # schema loc.
                                      nsmap={None: page_const.NS_PAGE_XML})  # Default ns

        metadata = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sMETADATA_ELT))
        xml_page_root.append(metadata)
        creator = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCREATOR_ELT))
        creator.text = creator_name
        created = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sCREATED_ELT))
        created.text = datetime.datetime.utcnow().isoformat() + "Z"
        last_change = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, page_const.sLAST_CHANGE_ELT))
        last_change.text = datetime.datetime.utcnow().isoformat() + "Z"
        metadata.append(creator)
        metadata.append(created)
        metadata.append(last_change)

        page_node = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, 'Page'))
        page_node.set('imageFilename', filename)
        page_node.set('imageWidth', str(img_w))
        page_node.set('imageHeight', str(img_h))

        xml_page_root.append(page_node)

        page_doc = etree.ElementTree(xml_page_root)
        b_validate = self.validate(page_doc)
        assert b_validate, 'new file not validated by schema'

        return page_doc

    @classmethod
    def create_page_xml_node(cls, node_name):
        """
            create a PageXMl element
        """
        node = etree.Element('{%s}%s' % (page_const.NS_PAGE_XML, node_name))

        return node

    def remove_page_xml_node(cls, nd: etree.ElementBase):
        """
            remove a PageXml element
        """
        nd.getparent().remove(nd)

    def insert_page_xml_node(self, parent_nd, node_name):
        """ Add PageXml node as child node of ``parent_nd``.

        :param parent_nd: node where PageXml node is added as child
        :param node_name: name of the node
        :return: the inserted node
        """
        node = self.create_page_xml_node(node_name)
        parent_nd.append(node)

        return node

    def load_page_xml(self, path_to_xml):
        """Load PageXml file located at ``path_to_xml`` and return a DOM node.

        :param path_to_xml: path to PageXml file
        :return: DOM document node
        :rtype: etree._ElementTree
        """
        page_doc = etree.parse(path_to_xml, etree.XMLParser(remove_blank_text=True))
        if not self.validate(page_doc):
            logger.debug(
                "PageXml is not valid according to the Page schema definition {}.".format(page_const.XSILOCATION))

        return page_doc

    def write_page_xml(self, save_path, creator=page_const.sCREATOR, comments=None):
        """Save PageXml file to ``save_path``.

        @:param save_path:
        @:return: None
        """
        self.set_metadata(creator, comments)

        with open(save_path, "w") as f:
            f.write(etree.tostring(self.page_doc, pretty_print=True, encoding="UTF-8", standalone=True,
                                   xml_declaration=True).decode("utf-8"))

    @staticmethod
    def _get_transkribus_meta_from_nd(nd_transkribus_meta):
        if nd_transkribus_meta is None:
            return None

        transkribus_meta = TranskribusMetadata(
            docId=nd_transkribus_meta.get('docId'),
            pageId=nd_transkribus_meta.get('pageId'),
            pageNr=nd_transkribus_meta.get('pageNr'),
            tsid=nd_transkribus_meta.get('tsid'),
            status=nd_transkribus_meta.get('status'),
            userId=nd_transkribus_meta.get('userId'),
            imgUrl=nd_transkribus_meta.get('imgUrl'),
            xmlUrl=nd_transkribus_meta.get('xmlUrl'),
            imageId=nd_transkribus_meta.get('imageId')
        )

        return transkribus_meta


# =========== METADATA OF PAGEXML ===========
class Metadata:
    """
    <complexType name="MetadataType">
        <sequence>
            <element name="Creator" type="string"></element>
            <element name="Created" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="LastChange" type="dateTime">
                <annotation>
                    <documentation>The timestamp has to be in UTC (Coordinated Universal Time) and not local time.</documentation></annotation></element>
            <element name="Comments" type="string" minOccurs="0"
                maxOccurs="1"></element>
        </sequence>
    </complexType>
    """

    def __init__(self, creator, created, last_change, comments=None, transkribus_meta=None):
        self.Creator = creator  # a string
        self.Created = created  # a string
        self.LastChange = last_change  # a string
        self.Comments = comments  # None or a string
        self.TranskribusMeta = transkribus_meta # None or TranskribusMetadata object

class TranskribusMetadata:
    """
    <complexType name="TranskribusMetadataType">
        <attribute name="docId" type="string"/>
        <attribute name="pageId" type="string"/>
        <attribute name="pageNr" type="integer"/>
        <attribute name="tsid" type="string"/>
        <attribute name="status" type="string"/>
        <attribute name="userId" type="string"/>
        <attribute name="imgUrl" type="anyURI"/>
        <attribute name="xmlUrl" type="anyURI"/>
        <attribute name="imageId" type="string"/>
    </complexType>
    """

    def __init__(self, docId=None, pageId=None, pageNr=None, tsid=None, status=None, userId=None, imgUrl=None,
                 xmlUrl=None, imageId=None):
        self.docId = docId
        self.pageId = pageId
        self.pageNr = pageNr
        self.tsid = tsid
        self.status = status
        self.userId = userId
        self.imgUrl = imgUrl
        self.xmlUrl = xmlUrl
        self.imageId = imageId



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path_to_xml', default='', type=str, metavar="STR",
                        help="path to the PageXml file")
    flags = parser.parse_args()

    path_to_xml = flags.path_to_xml
    if path_to_xml == '':
        path_to_xml = "/home/max/data/newseye/gt_data/trnskrbs_61381_nlf/330110/1895_06_01/page/330110_0001_13011771_old.xml"

    print(path_to_xml)

    page = Page(path_to_xml)

    transkribus_metadata: TranskribusMetadata = page.metadata.TranskribusMeta
    print("imageId = ", transkribus_metadata.imageId)

    # text_region = page.get_text_regions()[1]
    # print("TextRegion: ", text_region)
    # page.set_text_regions([text_region], overwrite=True)
    # new_text_region = page.get_text_regions()[0]
    # print(new_text_region.custom)

    # print(page.get_article_dict())

    # textlines = page.get_textlines()
    # for tl in textlines:
    #     if tl.get_article_id() is not None:
    #         tl.set_article_id(None)
    # page.set_textline_attr(textlines)

    # page.write_page_xml("./test/resources/page_xml_no_meta_copy.xml")

    # textlines = page.get_textlines()
    # # set all textline article ids to "a1"
    # # textline attrs are changed via id -> for now adding textlines is not supported!
    # for tl in textlines:
    #     tl.set_article_id("a1")
    #     # print(tl.baseline.points_list)
    #     print(tl.surr_p.to_polygon().x_points)
    # page.set_textline_attr(textlines)
    # page.write_page_xml("./test/resources/page_xml_copy.xml")
