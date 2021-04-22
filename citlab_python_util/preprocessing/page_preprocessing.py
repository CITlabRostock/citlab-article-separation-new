import os
from pathlib import Path
from shutil import copyfile

import citlab_python_util.parser.xml.page.page as page
from citlab_python_util.basic.list_util import filter_by_attribute
from citlab_python_util.io import file_loader

BATCH_SIZE = 100


def batch(iterable, batch_size=1):
    iterable_length = len(iterable)
    for i in range(0, iterable_length, batch_size):
        yield iterable[i:min(i + batch_size, iterable_length)]


class PagePreProcessor:
    """
    PagePreProcessor is a tool that corrects PAGE-XML files, e.g. deleting PAGE objects with the same ID.
    """

    def __init__(self, page_path_list):
        self.page_path_list_full = file_loader.load_text_file(page_path_list)
        self.num_files = len(self.page_path_list_full)
        self.page_path_list = [x for x in batch(self.page_path_list_full, BATCH_SIZE)]
        self.current_batch_idx = 0
        self.num_batches = len(self.page_path_list)
        self.page_object_list = self.create_page_objects(batch_idx=self.current_batch_idx)

    def update_current_batch_idx(self):
        self.current_batch_idx = min(self.num_batches, self.current_batch_idx + 1)

    def update_step(self):
        self.update_current_batch_idx()
        self.page_object_list = self.create_page_objects(self.current_batch_idx)

    def create_page_objects(self, batch_idx):
        return [page.Page(path_to_page) for path_to_page in self.page_path_list[batch_idx]]

    def delete_textlines_with_same_id(self):
        print(f"Start deleting redundant text lines for batch {self.current_batch_idx}..")
        for i, page_object in enumerate(self.page_object_list):
            textlines = page_object.get_textlines(ignore_redundant_textlines=False)
            if len(textlines) == 0:
                print(
                    f"{int((i + 1) / len(self.page_object_list) * 100):>3}%: Found no text lines in page file "
                    f"'{self.page_path_list[self.current_batch_idx][i]}'")
                continue

            tl_id_dict = filter_by_attribute(textlines, "id")
            redundant_textline_count = 0
            for tl_id, tl_list in tl_id_dict.items():
                if len(tl_list) > 1:
                    redundant_textline_count += 1
                    nds = page_object.get_child_by_id(page_object.page_doc, tl_id)
                    for nd in nds[1:]:
                        page_object.remove_page_xml_node(nd)
            print(
                f"{int((i + 1) / len(self.page_object_list) * 100):>3}%: Found {redundant_textline_count} text line ids with multiple"
                f" assigned text lines in page file '{self.page_path_list[self.current_batch_idx][i]}'")

    def delete_border_textlines(self):
        print(
            f"Start deleting text lines not belonging to the main page (i.e., covering the margin of neighboring pages"
            f" due to bad scans) for batch {self.current_batch_idx}..")
        import time
        start = time.time()
        for i, page_object in enumerate(self.page_object_list):
            textlines = page_object.get_textlines()
            if len(textlines) == 0:
                print(
                    f"{int((i + 1) / len(self.page_object_list) * 100):>3}%: Found no text lines in page file "
                    f"'{self.page_path_list[self.current_batch_idx][i]}'")
                continue

            # sort text lines by the x-values (small to big)
            textlines_from_left = sorted(textlines, key=lambda textline: min(textline.baseline.to_polygon().x_points))
            textlines_from_right = sorted(textlines, key=lambda textline: max(textline.baseline.to_polygon().x_points))
            # textlines = sorted(textlines, key=lambda textline: textline.baseline.points_list[0][0])

            # Get the average textline length across the page
            # avg_baseline_length = sum(
            #     [max(textline.baseline.to_polygon().x_points) - min(textline.baseline.to_polygon().x_points) for
            #      textline in textlines])/len(textlines)
            baseline_lengths_left = [max(textline.baseline.to_polygon().x_points) - min(textline.baseline.to_polygon().x_points) for textline in textlines_from_left]
            baseline_lengths_right = [max(textline.baseline.to_polygon().x_points) - min(textline.baseline.to_polygon().x_points) for textline in textlines_from_right]
            avg_baseline_length = sum(baseline_lengths_left)/len(textlines)

            count_removed_textlines = 0
            min_margin = 80
            min_start_x = min_margin
            for textline_l, baseline_length_l in zip(textlines_from_left, baseline_lengths_left):
                textline_start_x = min(textline_l.baseline.to_polygon().x_points)
                if textline_start_x >= min_start_x:
                    break
                if baseline_length_l < avg_baseline_length / 2:
                    textline_nd = page_object.get_child_by_id(page_object.page_doc, textline_l.id)[0]
                    page_object.remove_page_xml_node(textline_nd)
                    count_removed_textlines += 1

            max_end_x = page_object.get_image_resolution()[0] - min_margin
            for textline_r, baseline_length_r in zip(textlines_from_right, baseline_lengths_right):
                textline_end_x = max(textline_r.baseline.to_polygon().x_points)
                if textline_end_x <= max_end_x:
                    break
                if baseline_length_r < avg_baseline_length / 2:
                    textline_nd = page_object.get_child_by_id(page_object.page_doc, textline_r.id)[0]
                    page_object.remove_page_xml_node(textline_nd)
                    count_removed_textlines += 1

            print(f"Removed {count_removed_textlines}")
        print(f"Took {time.time() - start}")

    def save_page_files(self, overwrite=False, save_folder=None):
        """
            Saving the (modified) page files coming from the preprocessor. There are four cases for the tuple (`overwrite`, `save_folder`):
            - (True, None) or (True, path): overwrite the (modified) page files
            - (False, None): backup the loaded page file (just append a '.bak') before saving the modified version
            - (False, path): save the (modified) page file to the directory given by save_folder
        """
        common_prefix = ""
        if save_folder:
            common_prefix = os.path.dirname(os.path.commonprefix(self.page_path_list_full)) + os.path.sep
        for page_path, page_object in zip(self.page_path_list[self.current_batch_idx], self.page_object_list):
            page_path_folder = os.path.dirname(page_path)
            # abs_save_folder = os.path.abspath(save_folder)
            # abs_page_path_folder = os.path.abspath(page_path_folder)
            real_save_folder = os.path.realpath(save_folder) if save_folder is not None else None
            real_page_path_folder = os.path.realpath(page_path_folder)

            if not overwrite and (save_folder is None or real_save_folder == real_page_path_folder):
                save_path = page_path
                copyfile(page_path, page_path + '.bak')
            elif overwrite or save_folder is None or real_save_folder == real_page_path_folder:
                save_path = page_path
            else:
                page_suffix = page_path.split(common_prefix)[-1]
                save_path = os.path.join(save_folder, page_suffix)
                if save_path == page_path:
                    raise ValueError("This behavior should not occur! "
                                     "If the save folder is equal to the path where the page is stored, "
                                     "the file should be backed up.")
                path = Path(os.path.dirname(save_path))
                path.mkdir(parents=True, exist_ok=True)

            page_object.write_page_xml(save_path)


if __name__ == '__main__':
    page_path_list = "/home/max/data/newseye/pipeline_datasets/nyh_data/12148_bpt6k4118689m/page_files.lst"
    page_pre_proc = PagePreProcessor(page_path_list)
    page_pre_proc.delete_border_textlines()
