import argparse

from citlab_article_separation.ground_truth_generators.region_ground_truth_generator import \
    RegionGroundTruthGenerator


class BNLGroundTruthGeneratorHeaders(RegionGroundTruthGenerator):
    def __init__(self, path_to_img_lst, max_resolution=(0, 0), scaling_factor=1.0, use_bounding_box=False,
                 use_min_area_rect=False, issue_name="luxwort"):
        super().__init__(path_to_img_lst, max_resolution, scaling_factor, use_bounding_box, use_min_area_rect)
        self.issue_name = issue_name

        title_headline_regions = self.get_title_regions_list(["headline"])
        title_subheadline_regions = self.get_title_regions_list(["subheadline", "motto"])
        title_other_regions = self.get_title_regions_list(["other", "publishing_stmt"])

        heading_title_regions = self.get_classic_heading_regions_list(["", "title", "subheadline", "overline"])

        # Make sure that the order of items in the dictionary is the same as the order of the GT files below
        self.TITLE_HEADLINE_REGIONS = "title_headline_regions"
        self.TITLE_SUBHEADLINE_REGIONS = "title_subheadline_regions"
        self.TITLE_OTHER_REGIONS = "title_other_regions"
        self.HEADING_TITLE_REGIONS = "heading_title_regions"

        if self.issue_name == "independance_lux":
            self.regions_dict = {self.TITLE_HEADLINE_REGIONS: title_headline_regions,
                                 self.TITLE_OTHER_REGIONS: title_other_regions,
                                 self.HEADING_TITLE_REGIONS: heading_title_regions}
        else:
            self.regions_dict = {self.TITLE_HEADLINE_REGIONS: title_headline_regions,
                                 self.TITLE_SUBHEADLINE_REGIONS: title_subheadline_regions,
                                 self.TITLE_OTHER_REGIONS: title_other_regions,
                                 self.HEADING_TITLE_REGIONS: heading_title_regions}

    def create_ground_truth_images(self):
        # Order of gt images is important for the "make_disjoint_all()" call at the end.
        for i in range(len(self.img_path_lst)):
            print(f"{i + 1}/{len(self.img_path_lst)}: {self.img_path_lst[i]}")
            img_width = self.img_res_lst[i][1]
            img_height = self.img_res_lst[i][0]
            sc_factor = self.scaling_factors[i]

            if all(len(regions[i]) == 0 for regions in self.regions_dict.values()):
                print("\tSkipping because requested GT is not available on this page.")
                continue

            self.gt_dict[self.TITLE_HEADLINE_REGIONS].append(self.create_region_gt_img(
                self.regions_dict[self.TITLE_HEADLINE_REGIONS][i], img_width, img_height, fill=True,
                scaling_factor=sc_factor))
            self.gt_dict[self.TITLE_OTHER_REGIONS].append(self.create_region_gt_img(
                self.regions_dict[self.TITLE_OTHER_REGIONS][i], img_width, img_height, fill=True,
                scaling_factor=sc_factor))
            self.gt_dict[self.HEADING_TITLE_REGIONS].append(self.create_region_gt_img(
                self.regions_dict[self.HEADING_TITLE_REGIONS][i], img_width, img_height, fill=True,
                scaling_factor=sc_factor))

            if self.issue_name == "independance_lux":
                gt_channels = [self.gt_dict[self.TITLE_HEADLINE_REGIONS][-1],
                               self.gt_dict[self.TITLE_OTHER_REGIONS][-1],
                               self.gt_dict[self.HEADING_TITLE_REGIONS][-1]]

            else:
                self.gt_dict[self.TITLE_SUBHEADLINE_REGIONS].append(self.create_region_gt_img(
                    self.regions_dict[self.TITLE_SUBHEADLINE_REGIONS][i], img_width, img_height, fill=True,
                    scaling_factor=sc_factor))

                gt_channels = [self.gt_dict[self.TITLE_HEADLINE_REGIONS][-1],
                               self.gt_dict[self.TITLE_SUBHEADLINE_REGIONS][-1],
                               self.gt_dict[self.TITLE_OTHER_REGIONS][-1],
                               self.gt_dict[self.HEADING_TITLE_REGIONS][-1]]

            other_gt_img = self.create_other_ground_truth_image(*gt_channels)
            gt_channels.append(other_gt_img)
            gt_channels = tuple(gt_channels)

            self.gt_imgs_lst.append(gt_channels)
            self.valid_img_indizes.append(i)
        self.make_disjoint_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--max_height', type=int, default=0)
    parser.add_argument('--max_width', type=int, default=0)
    parser.add_argument('--scaling_factor', type=float, default=1.0)
    parser.add_argument('--newspaper_issue', type=str, choices=['luxwort', 'independance_lux'])
    parser.add_argument('--save_info_file', type=bool, default=True)

    args = parser.parse_args()

    tb_generator = BNLGroundTruthGeneratorHeaders(
        args.image_list, use_bounding_box=False, use_min_area_rect=False,
        max_resolution=(args.max_height, args.max_width), scaling_factor=args.scaling_factor,
        issue_name=args.newspaper_issue)

    tb_generator.run_ground_truth_generation(args.save_dir, args.save_info_file)
