import os
import re
import glob


def get_path_from_exportdir(model_dir, pattern, not_pattern):
    export_dir = os.path.join(model_dir, "export")
    name = [x for x in glob.glob1(export_dir, pattern) if not_pattern not in x]
    if len(name) == 1:
        return os.path.join(export_dir, name[0])
    else:
        raise IOError(f"Found {len(name)} '{pattern}' files in {export_dir}, there must be exact one.")


def get_img_from_page_path(page_path):
    # go up the page folder, remove .xml ending and check for img file
    img_endings = ("tif", "jpg", "png")
    img_path = re.sub(r'/page/([-\w.]+)\.xml$', r'/\1', page_path)
    for ending in img_endings:
        if img_path.endswith(ending):
            if os.path.isfile(img_path):
                return img_path
    # go up the page folder, substitute .xml ending and check for img file
    img_path = re.sub(r'/page/([-\w.]+)\.xml$', r'/\1.tif', page_path)
    if not os.path.isfile(img_path):
        img_path = re.sub(r'tif$', r'png', img_path)
        if not os.path.isfile(img_path):
            img_path = re.sub(r'png$', r'jpg', img_path)
            if not os.path.isfile(img_path):
                raise IOError(f"No image file (tif, png, jpg) found to given pagexml {page_path}")
    return img_path


def get_img_from_json_path(json_path):
    # go up the json folder, remove .json ending and check for img file
    img_endings = ("tif", "jpg", "png")
    img_path = re.sub(r'/json\w*/([-\w.]+)\.json$', r'/\1', json_path)
    for ending in img_endings:
        if img_path.endswith(ending):
            if os.path.isfile(img_path):
                return img_path
    # go up the json folder, substitute .json ending and check for img file
    img_path = re.sub(r'/json\w*/([-\w.]+)\.json$', r'/\1.tif', json_path)
    if not os.path.isfile(img_path):
        img_path = re.sub(r'tif$', r'png', img_path)
        if not os.path.isfile(img_path):
            img_path = re.sub(r'png$', r'jpg', img_path)
            if not os.path.isfile(img_path):
                raise IOError("No image file (tif, png, jpg) found to given json ", json_path)
    return img_path


def get_page_from_img_path(img_path):
    # go into page folder, append .xml and check for pageXML file
    page_path = re.sub(r'/([-\w.]+)$', r'/page/\1.xml', img_path)
    if os.path.isfile(page_path):
        return page_path
    # go into page folder, substitute img ending for .xml and check for pageXML file
    page_path = re.sub(r'/([-\w.]+)\.\w+$', r'/page/\1.xml', img_path)
    if not os.path.isfile(page_path):
        raise IOError("No pagexml file found to given img file ", img_path)
    return page_path


def get_page_from_json_path(json_path):
    # go into page folder, append .xml and check for pageXML file
    page_path = re.sub(r'/json\w*/([-\w.]+)$', r'/page/\1.xml', json_path)
    if os.path.isfile(page_path):
        return page_path
    # go into page folder, substitute .json for .xml and check for pageXML file
    page_path = re.sub(r'/json\w*/([-\w.]+)\.json$', r'/page/\1.xml', json_path)
    if not os.path.isfile(page_path):
        raise IOError("No pagexml file found to given json file ", json_path)
    return page_path


def get_page_from_conf_path(json_path):
    page_path = re.sub(r'/confidences/([-\w.]+)_confidences\.json$', r'/page/\1.xml', json_path)
    if not os.path.isfile(page_path):
        raise IOError("No pagexml file found to given (confidence) json file ", json_path)
    return page_path


def prepend_folder_name(file_path):
    folder_path = os.path.dirname(file_path)
    folder_name = os.path.basename(folder_path)
    file_name = os.path.basename(file_path)
    new_file_name = folder_name + "_" + file_name

    return os.path.join(folder_path, new_file_name)
