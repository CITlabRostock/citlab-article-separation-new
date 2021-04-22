from copy import deepcopy

import numpy as np
from shapely import geometry, validation

from citlab_article_separation.net_post_processing.region_to_page_writer import RegionToPageWriter
from citlab_python_util.parser.xml.page.page_constants import sSEPARATORREGION, sTEXTREGION
from citlab_python_util.parser.xml.page.page_objects import SeparatorRegion


class SeparatorRegionToPageWriter(RegionToPageWriter):
    def __init__(self, path_to_page, path_to_image=None, fixed_height=None, scaling_factor=None, region_dict=None):
        super().__init__(path_to_page, path_to_image, fixed_height, scaling_factor)
        self.region_dict = region_dict

    def remove_separator_regions_from_page(self):
        self.page_object.remove_regions(sSEPARATORREGION)

    def convert_polygon_with_holes(self, polygon_sh):
        def split_horiz_by_point(polygon, point):
            """"""
            assert polygon.geom_type == "Polygon" and point.geom_type == "Point"
            nx, ny, xx, xy = polygon.bounds
            if point.x < nx or point.x > xx:
                return [polygon]

            lEnv = geometry.LineString([(nx, ny), (point.x, xy)]).envelope
            rEnv = geometry.LineString([(point.x, ny), (xx, xy)]).envelope

            try:
                return [polygon.intersection(lEnv), polygon.intersection(rEnv)]
            except Exception as e:
                print("Geometry error: %s" % validation.explain_validity(polygon))
                return [polygon.buffer(0)]

        parts = []
        if polygon_sh.type == "MultiPolygon":
            for p in polygon_sh.geoms:
                parts.extend(self.convert_polygon_with_holes(p))
        elif polygon_sh.type == "Polygon":
            if len(polygon_sh.interiors):
                pt = polygon_sh.interiors[0].centroid
                halves = split_horiz_by_point(polygon_sh, pt)
                for p in halves:
                    parts.extend(self.convert_polygon_with_holes(p))
            else:
                parts = [polygon_sh]

        return parts

    def convert_polygon_with_holes2(self, polygon_sh):
        def closest_pt(pt, ptset):
            """"""
            dist2 = np.sum((ptset - pt) ** 2, 1)
            minidx = np.argmin(dist2)
            return minidx

        def cw_perpendicular(pt, norm=None):
            """"""
            d = np.sqrt((pt ** 2).sum()) or 1
            if norm is None:
                return np.array([pt[1], -pt[0]])
            return np.array([pt[1], -pt[0]]) / d * norm

        def lazy_short_join_gap(exter, inter, refpt, gap=0.000001):
            """"""
            exIdx = closest_pt(refpt, exter)

            inIdx = closest_pt(exter[exIdx], inter)
            print(exter[exIdx], inter[inIdx])
            excwgap = exter[exIdx] + cw_perpendicular(inter[inIdx] - exter[exIdx], gap)
            incwgap = inter[inIdx] + cw_perpendicular(exter[exIdx] - inter[inIdx], gap)
            out = np.vstack((exter[:exIdx], excwgap, inter[inIdx:-1], inter[:inIdx], incwgap, exter[exIdx:]))
            out[-1] = out[0]

            return out

        if len(polygon_sh.interiors):
            ex = np.asarray(polygon_sh.exterior)
            for inter in polygon_sh.interiors:
                inArr = np.asarray(inter)
                ex = lazy_short_join_gap(ex, inArr, np.asarray(inter.centroid))
            poly = geometry.Polygon(ex)
            print(len(list(poly.interiors)))
        return poly

    def merge_regions(self, remove_holes=True):
        def _split_shapely_polygon(region_to_split_sh, region_compare_sh):
            # region_to_split_sh = region_to_split_sh.buffer(0)
            # region_compare_sh = region_compare_sh.buffer(0)
            difference = region_to_split_sh.difference(region_compare_sh)
            if type(difference) == geometry.MultiPolygon or type(difference) == geometry.MultiLineString:
                new_region_polys_sh = list(difference)
            else:
                new_region_polys_sh = [difference]

            return new_region_polys_sh

        def _create_page_objects(region_to_split, new_region_polys):
            new_region_objects = [deepcopy(region_to_split) for _ in range(len(new_region_polys))]

            for j, (new_region_poly, new_region_object) in enumerate(
                    zip(new_region_polys, new_region_objects)):
                new_region_object.set_points(new_region_poly)
                if len(new_region_polys) > 1:
                    new_region_object.id = region_to_split.id + "_" + str(j + 1)

            return new_region_objects

        def _delete_region_from_page(region_id):
            region_to_delete = self.page_object.get_child_by_id(self.page_object.page_doc, region_id)
            if len(region_to_delete) == 0:
                return
            region_to_delete = region_to_delete[0]
            self.page_object.remove_page_xml_node(region_to_delete)

        def _add_regions_to_page(region_object_list):
            for region_object in region_object_list:
                self.page_object.add_region(region_object)

        def _get_parent_region(child_split_sh, parent_splits_sh):
            for j, parent_split_sh in enumerate(parent_splits_sh):
                if child_split_sh.intersects(parent_split_sh):
                    return j, parent_split_sh
            return None, None

        def _split_text_lines(text_lines_dict, sep_poly):
            """
            Given a separator polygon `sep_poly` split just the text lines (and its baselines) given by
            `text_line_list`. `sep_poly` is a list of lists of polygon coordinates. If the separator polygon is only
            described via one exterior polygon, the list of lists has length 1. Otherwise, there are also inner
            polygons, i.e. the list of lists has a length > 1.
            :param text_line_list:
            :param sep_poly:
            :return:
            """
            sep_poly_sh = geometry.Polygon(sep_poly[0], sep_poly[1:]).buffer(0)
            if type(sep_poly_sh) == geometry.MultiPolygon:
                sep_poly_sh = sep_poly_sh[np.argmax([poly.area for poly in list(sep_poly_sh)])]

            for tl_id, text_lines in text_lines_dict.items():
                for text_line in text_lines:
                    text_line_sh = geometry.Polygon(text_line.surr_p.points_list).buffer(0)
                    # If text line is contained completely in the vertical separator polygon delete it
                    if sep_poly_sh.contains(text_line_sh):
                        text_lines_dict[tl_id].remove(text_line)
                        continue
                    if text_line_sh.intersects(sep_poly_sh):
                        text_line_splits_sh = _split_shapely_polygon(text_line_sh, sep_poly_sh)
                        text_line_splits = [list(poly.exterior.coords) for poly in text_line_splits_sh]

                        new_text_line_objects = _create_page_objects(text_line, text_line_splits)

                        for new_text_line_object in new_text_line_objects:
                            new_text_line_object.set_baseline(None)
                            if len(new_text_line_objects) != 1:
                                new_text_line_object.words = []

                        if len(new_text_line_objects) != 1:
                            for word in text_line.words:
                                # Assumes that the words are in the right order
                                word_polygon_sh = geometry.Polygon(word.surr_p.points_list).buffer(0)
                                matching_textline_idx = np.argmax([word_polygon_sh.intersection(text_line_split_sh).area
                                                                   for text_line_split_sh in text_line_splits_sh])
                                corr_textline = new_text_line_objects[matching_textline_idx]
                                corr_textline.words.append(word)

                            if len(text_line.words) > 0:
                                for new_text_line_object in new_text_line_objects:
                                    new_text_line_object.text = " ".join([word.text for word in new_text_line_object.words])

                        baseline_sh = geometry.LineString(
                            text_line.baseline.points_list) if text_line.baseline is not None else None
                        if baseline_sh is not None and baseline_sh.intersects(sep_poly_sh):
                            baseline_splits = _split_shapely_polygon(baseline_sh, sep_poly_sh)
                        elif baseline_sh is not None:
                            baseline_splits = [baseline_sh]

                        # baseline split -> text line split
                        used_idx = set()
                        for baseline_split in baseline_splits:
                            idx, parent_text_line = _get_parent_region(baseline_split,
                                                                       text_line_splits_sh)
                            if idx is None:
                                continue
                            used_idx.add(idx)
                            new_text_line_objects[idx].set_baseline(list(baseline_split.coords))

                        # Remove all text line splits that don't have an associated baseline split
                        # TODO: Maybe rather consider the word elements instead?
                        new_text_line_objects = [new_text_line_objects[idx] for idx in used_idx]
                        text_lines_dict[tl_id].extend(new_text_line_objects)
                        text_lines_dict[tl_id].remove(text_line)
            return text_lines_dict

        def _split_regions(region_dict, sep_poly):
            """
            Given a SeparatorRegion, split regions in region_dict if possible/necessary. Returns False if one of the
            regions in `region_dict` contains the SeparatorRegion. Then don't write it to the PAGE file.
            This function assumes, that the text lines lie completely within the text regions and the baselines lie
            completely within the text lines.
            :param region_dict:
            :param sep_poly:
            :return:
            """
            sep_poly_sh = geometry.Polygon(sep_poly).buffer(0)
            if type(sep_poly_sh) == geometry.MultiPolygon:
                sep_poly_sh = sep_poly_sh[np.argmax([poly.area for poly in list(sep_poly_sh)])]
                # sep_poly_sh = sep_poly_sh[max(range(len(list(sep_poly_sh))), key=lambda i: list(sep_poly_sh)[i].area)]

            for region_type, region_list in region_dict.items():
                updated_region_list = deepcopy(region_list)
                all_new_region_objects = []
                for i, region in enumerate(region_list):
                    region_polygon_sh = geometry.Polygon(region.points.points_list)
                    if region_polygon_sh.intersects(sep_poly_sh):
                        if region_polygon_sh.contains(sep_poly_sh) or sep_poly_sh.contains(region_polygon_sh):
                            # don't need to check the other regions, provided that we don't have overlapping regions
                            return False

                        new_region_polys_sh = _split_shapely_polygon(region_polygon_sh, sep_poly_sh)
                        new_region_polys = [list(poly.exterior.coords) for poly in new_region_polys_sh]
                        new_region_objects = _create_page_objects(region, new_region_polys)

                        # if the region is a TextRegion we also need to take care of the baselines and text lines
                        if region_type == sTEXTREGION:
                            for new_region_object in new_region_objects:
                                new_region_object.text_lines = []

                            text_lines = region.text_lines
                            for text_line in text_lines:
                                text_line_sh = geometry.Polygon(text_line.surr_p.points_list).buffer(0)
                                if sep_poly_sh.contains(text_line_sh):
                                    return False
                                if text_line_sh.intersects(sep_poly_sh):
                                    text_line_splits_sh = _split_shapely_polygon(text_line_sh, sep_poly_sh)
                                    text_line_splits = [list(poly.exterior.coords) for poly in text_line_splits_sh]

                                    new_text_line_objects = _create_page_objects(text_line, text_line_splits)
                                    for new_text_line_object in new_text_line_objects:
                                        new_text_line_object.set_baseline(None)
                                        new_text_line_object.words = []

                                    # word_idx = np.argmax(
                                    #     [geometry.Polygon(word.surr_p.points_list).buffer(0).distance(sep_poly_sh)
                                    #      for word in text_line.words])

                                    for word in text_line.words:
                                        word_polygon_sh = geometry.Polygon(word.surr_p.points_list).buffer(0)
                                        matching_textline_idx = np.argmax([word_polygon_sh.intersection(text_line_split_sh)
                                        for text_line_split_sh in text_line_splits_sh])
                                        corr_textline = new_text_line_objects[matching_textline_idx]
                                        corr_textline.words.append(word)

                                    if len(text_line.words) > 0:
                                        for new_text_line_object in new_text_line_objects:
                                            new_text_line_object.text = " ".join([word.text for word in text_line.words])

                                    baseline_sh = geometry.LineString(
                                        text_line.baseline.points_list) if text_line.baseline is not None else None
                                    if baseline_sh is not None and baseline_sh.intersects(sep_poly_sh):
                                        baseline_splits = _split_shapely_polygon(baseline_sh, sep_poly_sh)

                                        # baseline split -> text line split
                                        for baseline_split in baseline_splits:
                                            idx, parent_text_line = _get_parent_region(baseline_split,
                                                                                       text_line_splits_sh)
                                            if idx is None:
                                                continue
                                            new_text_line_objects[idx].set_baseline(list(baseline_split.coords))

                                else:
                                    text_line_splits_sh = [text_line_sh]
                                    new_text_line_objects = [text_line]

                                # text line split -> region split
                                for text_line_split, new_text_line_object in zip(text_line_splits_sh,
                                                                                 new_text_line_objects):
                                    idx, parent_region = _get_parent_region(text_line_split, new_region_polys_sh)
                                    if idx is None:
                                        continue
                                    new_region_objects[idx].text_lines.append(new_text_line_object)

                        _delete_region_from_page(region.id)
                        offset = len(region_list) - len(updated_region_list)
                        updated_region_list.pop(i - offset)

                        # updated_region_list[i:i + 1] = new_region_objects

                        all_new_region_objects.extend(new_region_objects)
                        _add_regions_to_page(new_region_objects)
                updated_region_list.extend(all_new_region_objects)
                region_dict[region_type] = updated_region_list

                # _add_regions_to_page(all_new_region_objects)

                return True

        def _add_separator_regions_to_page(separator_polygons, remove_holes=False):
            for separator_polygon in separator_polygons:
                if remove_holes and len(separator_polygon) > 1:
                    separator_polygon_ext = separator_polygon[0]
                    separator_polygon_int = separator_polygon[1:]
                    separator_polygon_int = [int_poly for int_poly in separator_polygon_int
                                             if geometry.Polygon(int_poly).area > 1000]
                    separator_polygon_sh = geometry.Polygon(separator_polygon_ext, separator_polygon_int).buffer(0)

                    separator_polygon_parts_sh = self.convert_polygon_with_holes(separator_polygon_sh)
                    separator_polygon_parts = [list(sep_part.exterior.coords) for sep_part in separator_polygon_parts_sh]
                    # separator_polygon = list(separator_polygon_sh.exterior.coords)

                    for separator_polygon_part in separator_polygon_parts:
                        separator_id = self.page_object.get_unique_id(sSEPARATORREGION)

                        custom_tag_dict = None
                        if separator_type != sSEPARATORREGION:
                            custom_tag_dict = {"structure": {"orientation": separator_type.lstrip(sSEPARATORREGION + "_")}}
                        separator_region = SeparatorRegion(separator_id, points=separator_polygon_part,
                                                           custom=custom_tag_dict)
                        self.page_object.add_region(separator_region)

                else:
                    # Ignore the inner polygons and only write the outer ones
                    separator_polygon = separator_polygon[0]
                    separator_id = self.page_object.get_unique_id(sSEPARATORREGION)

                    custom_tag_dict = None
                    if separator_type != sSEPARATORREGION:
                        custom_tag_dict = {"structure": {"orientation": separator_type.lstrip(sSEPARATORREGION + "_")}}
                    separator_region = SeparatorRegion(separator_id, points=separator_polygon, custom=custom_tag_dict)
                    self.page_object.add_region(separator_region)

        text_regions = self.page_object.get_text_regions()

        # For now we are only interested in the SeparatorRegion information
        for separator_type in [sSEPARATORREGION, sSEPARATORREGION + "_horizontal", sSEPARATORREGION + "_vertical"]:
            try:
                separator_polygons = self.region_dict[separator_type]
            except KeyError:
                continue

            if separator_type == sSEPARATORREGION + "_vertical":
                for text_region in text_regions:
                    # For each text line we remember its splits, initialize with the original baseline
                    text_lines_dict = {tl.id: [tl] for tl in text_region.text_lines}
                    for i, separator_polygon in enumerate(separator_polygons):
                        text_lines_dict = _split_text_lines(text_lines_dict, separator_polygon)

                    # TODO: Sort Text lines for an ID according to the x values
                    final_text_lines = []
                    for text_lines in text_lines_dict.values():
                        final_text_lines.extend(text_lines)
                    text_region.text_lines = final_text_lines
                    # self.page_object.set_text_lines(text_region, final_text_lines, overwrite=True)

                self.page_object.set_text_regions(text_regions, overwrite=True)

            # Add separator polygons to the page file.
            _add_separator_regions_to_page(separator_polygons, remove_holes)

