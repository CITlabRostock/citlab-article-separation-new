import numpy as np
import cv2


class StrokeWidthDistanceTransform(object):
    def __init__(self,
                 dark_on_bright=True,
                 clean_ccs=2):
        self._dark_on_bright = dark_on_bright
        self._clean_ccs = clean_ccs

    def apply_swt_dist_trafo(self, img_file):
        swt_dist_trafo = self.distance_transform(img_file)
        cc_boxes = self.connected_components_cv(swt_dist_trafo)
        cc_clean = self.clean_connected_components(cc_boxes)
        return swt_dist_trafo, cc_clean

    def distance_transform(self, img_file, norm=cv2.DIST_L2, mask=cv2.DIST_MASK_PRECISE):
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if self._dark_on_bright:
            image = -image + 255  # invert black/white
        threshold, image = self.otsu_threshold(image)  # binarize (otsu)
        dist_trafo = cv2.distanceTransform(image, norm, mask)
        return dist_trafo.astype(np.uint8)

    def otsu_threshold(self, image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        threshold, image_t = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold, image_t

    def connected_components_cv(self, image, connectivity=8):
        assert connectivity in (4, 8), f"Connectivity has to be 4 or 8 (was {connectivity})."
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
        boxes_ccs = []
        # start at 1, to skip background
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes_ccs.append((x, y, w, h))
        return boxes_ccs

    def clean_connected_components(self, components):
        components_clean = []
        # count_rejected_1 = 0
        # count_rejected_2 = 0
        for component in components:
            # component is a 4-tuple (x, y, width, height)
            width = component[2]
            height = component[3]
            if self._clean_ccs > 0:
                # test 1: reject components whose size is too small or too large
                if width < 3 or height < 3 or height > 500 or width > 500:
                    # count_rejected_1 += 1
                    continue
            if self._clean_ccs > 1:
                # test 2: reject components with too extreme aspect ratios (long narrow components)
                if width / height > 8 or height / width > 8:
                    # count_rejected_2 += 1
                    continue
            # component is accepted
            components_clean.append(component)
        # print(f"Rejected {count_rejected_1 + count_rejected_2}/{len(components)} connected components due to "
        #       f"cleaning (Size test: {count_rejected_1}, Ratio test: {count_rejected_2}).")
        return components_clean


if __name__ == '__main__':
    img_path = "abc"
    page_path = "xyz"

    from citlab_python_util.parser.xml.page.page import Page
    # Load page and textlines
    page = Page(page_path)
    text_lines = page.get_textlines()

    # initialize SWT and textline labeling
    SWT = StrokeWidthDistanceTransform(dark_on_bright=True)
    swt_img = SWT.distance_transform(img_path)
    textline_stroke_widths = dict()  # stroke widths for every text line
    textline_heights = dict()  # text height for every text line
    for text_line in text_lines:
        # build surrounding polygons over text lines
        bounding_box = text_line.surr_p.to_polygon().get_bounding_box()
        xa, xb = bounding_box.x, bounding_box.x + bounding_box.width
        ya, yb = bounding_box.y, bounding_box.y + bounding_box.height
        # get swt for text line
        text_line_swt = swt_img[ya:yb + 1, xa:xb + 1]
        # get connected components in text line
        text_line_ccs = SWT.connected_components_cv(text_line_swt)
        text_line_ccs = SWT.clean_connected_components(text_line_ccs)
        # go over connected components to estimate stroke width and text height of the text line
        swt_cc_values = []
        text_line_height = 0
        for cc in text_line_ccs:
            # component is a 4-tuple (x, y, width, height)
            # take max value in distance_transform as stroke_width for current CC (can be 0)
            swt_cc_values.append(np.max(text_line_swt[cc[1]: cc[1] + cc[3], cc[0]: cc[0] + cc[2]]))
            # new text height
            if cc[3] > text_line_height:
                text_line_height = cc[3]

        textline_stroke_widths[text_line.id] = np.median(swt_cc_values) if swt_cc_values else 0.0
        textline_heights[text_line.id] = text_line_height