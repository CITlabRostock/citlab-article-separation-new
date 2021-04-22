from argparse import ArgumentParser

import citlab_python_util.preprocessing.page_preprocessing as page_preprocessing

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--page_path_list', default='', type=str, metavar="STR",
                        help="path to the lst file containing the file paths of the PageXmls")
    parser.add_argument('--save_folder', default=None, type=str, metavar="STR",
                        help="save folder where files are stored. If not given, it saves the page files in place.")
    parser.add_argument('--overwrite', default=False, type=bool,
                        help="If true, it overwrites the page xml files modified by the preprocessor. "
                             "Defaults to False.")

    flags = parser.parse_args()
    page_path_list = flags.page_path_list
    save_folder = flags.save_folder
    overwrite = flags.overwrite

    user_input = input(f"Are these arguments correct?\n"
                       f"\tpage_path_list: {page_path_list}\n"
                       f"\tsave_folder: {save_folder}\n"
                       f"\toverwrite: {overwrite}\n[y/n]:")
    if user_input.upper() not in ["Y", "YES"]:
        print("Stopping.")
        exit(1)

    page_preprocessor = page_preprocessing.PagePreProcessor(page_path_list)
    for i in range(page_preprocessor.num_batches):
        page_preprocessor.delete_textlines_with_same_id()
        page_preprocessor.save_page_files(overwrite, save_folder)
        if i < page_preprocessor.num_batches - 1:
            page_preprocessor.update_step()