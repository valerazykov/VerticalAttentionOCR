#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import os
import shutil
import xml.etree.ElementTree as ET
import tarfile, zipfile
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, List
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_IAM_line():
    """
    Format the IAM dataset at line level with the commonly used split (6,482 for train, 976 for validation and 2,915 for test)
    """
    source_folder = "raw/IAM"
    target_folder = "formatted/IAM_lines"
    tar_filename = "lines.tgz"
    line_folder_path = os.path.join(target_folder, "lines")

    tar_path = os.path.join(source_folder, tar_filename)
    if not os.path.isfile(tar_path):
        print("error - {} not found".format(tar_path))
        exit(-1)

    os.makedirs(target_folder, exist_ok=True)
    tar = tarfile.open(tar_path)
    tar.extractall(line_folder_path)
    tar.close()

    set_names = ["train", "valid", "test"]
    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for set_name in set_names:
        id = 0
        current_folder = os.path.join(target_folder, set_name)
        os.makedirs(current_folder, exist_ok=True)
        xml_path = os.path.join(source_folder, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_fold_path = os.path.join(line_folder_path, name.split("-")[0], name)
            img_paths = [os.path.join(img_fold_path, p) for p in sorted(os.listdir(img_fold_path))]
            for i, line in enumerate(page[2]):
                label = line.attrib.get("Value")
                img_name = "{}_{}.png".format(set_name, id)
                gt[set_name][img_name] = {
                    "text": label,
                }
                charset = charset.union(set(label))
                new_path = os.path.join(current_folder, img_name)
                os.replace(img_paths[i], new_path)
                id += 1

    shutil.rmtree(line_folder_path)
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)


def format_IAM_paragraph():
    """
    Format the IAM dataset at paragraph level with the commonly used split (747 for train, 116 for validation and 336 for test)
    """
    source_folder = "raw/IAM"
    target_folder = "formatted/IAM_paragraph"
    img_folder_path = os.path.join(target_folder, "images")

    os.makedirs(target_folder, exist_ok=True)

    tar_filenames = ["formsA-D.tgz", "formsE-H.tgz", "formsI-Z.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(img_folder_path)
        tar.close()

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for set_name in ["train", "valid", "test"]:
        new_folder = os.path.join(target_folder, set_name)
        os.makedirs(new_folder, exist_ok=True)
        xml_path = os.path.join(source_folder, "{}.xml".format(set_name))
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_path = os.path.join(img_folder_path, name + ".png")
            new_name = "{}_{}.png".format(set_name, len(os.listdir(new_folder)))
            new_img_path = os.path.join(new_folder, new_name)
            lines = []
            full_text = ""
            for section in page:
                if section.tag != "Paragraph":
                    continue
                p_left, p_right = int(section.attrib.get("Left")), int(section.attrib.get("Right"))
                p_bottom, p_top = int(section.attrib.get("Bottom")), int(section.attrib.get("Top"))
                for i, line in enumerate(section):
                    words = []
                    for word in line:
                        words.append({
                            "text": word.attrib.get("Value"),
                            "left": int(word.attrib.get("Left")) - p_left,
                            "bottom": int(word.attrib.get("Bottom")) - p_top,
                            "right": int(word.attrib.get("Right")) - p_left,
                            "top": int(word.attrib.get("Top")) - p_top
                        })
                    lines.append({
                        "text": line.attrib.get("Value"),
                        "left": int(line.attrib.get("Left")) - p_left,
                        "bottom": int(line.attrib.get("Bottom")) - p_top,
                        "right": int(line.attrib.get("Right")) - p_left,
                        "top": int(line.attrib.get("Top")) - p_top,
                        "words": words
                    })
                    full_text = "{}{}\n".format(full_text, lines[-1]["text"])
                paragraph = {
                    "text": full_text[:-1],
                    "lines": lines
                }
                gt[set_name][new_name] = paragraph
                charset = charset.union(set(full_text))

                with Image.open(img_path) as pil_img:
                    img = np.array(pil_img)
                img = img[p_top:p_bottom + 1, p_left:p_right + 1]
                Image.fromarray(img).save(new_img_path)

    shutil.rmtree(img_folder_path)
    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)


def format_RIMES_line():
    """
    Format the RIMES dataset at line level with the commonly used split (the last 100 paragraph training samples are taken for the valid set)
    resulting in: 10,532 for training, 801 for validation and 778 for test.
    """
    source_folder = "raw/RIMES"
    target_folder = "formatted/RIMES_lines"
    img_folder_path = os.path.join(target_folder, "images_gray")
    os.makedirs(target_folder, exist_ok=True)

    eval_xml_path = os.path.join(source_folder, "eval_2011_annotated.xml")
    train_xml_path = os.path.join(source_folder, "training_2011.xml")
    for label_path in [eval_xml_path, train_xml_path]:
        if not os.path.isfile(label_path):
            print("error - {} not found".format(label_path))
            exit(-1)

    tar_filenames = ["eval_2011_gray.tar", "training_2011_gray.tar"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    nb_training_samples = sum([1 if "train" in name else 0 for name in os.listdir(img_folder_path)])
    nb_valid_samples = 100
    begin_valid_ind = nb_training_samples - nb_valid_samples

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    for set_name in gt.keys():
        os.makedirs(os.path.join(target_folder, set_name), exist_ok=True)

    charset = set()
    for set_name, xml_path in zip(["train", "eval"], [train_xml_path, eval_xml_path]):
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_path = os.path.join(img_folder_path, name+".png")
            img = np.array(Image.open(img_path, "r"))
            if set_name == "train":
                new_set_name = "train" if int(name.split("-")[-1])<begin_valid_ind else "valid"
            else:
                new_set_name = "test"
            fold = os.path.join(target_folder, new_set_name)
            for i, line in enumerate(page[0]):
                label = line.attrib.get("Value")
                new_name = "{}_{}.png".format(new_set_name, len(os.listdir(fold)))
                new_path = os.path.join(fold, new_name)
                left = max(0, int(line.attrib.get("Left")))
                bottom = int(line.attrib.get("Bottom"))
                right = int(line.attrib.get("Right"))
                top = max(0, int(line.attrib.get("Top")))
                line_img = img[top:bottom, left:right]
                try:
                    Image.fromarray(line_img).save(new_path)
                except:
                    print("problem")
                paragraph = {
                    "text": label,
                }
                charset = charset.union(set(label))
                gt[new_set_name][new_name] = paragraph

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)

    shutil.rmtree(img_folder_path)


def format_RIMES_paragraph():
    """
    Format the RIMES dataset at paragraph level with the official split (the last 100 training samples are taken for the valid set)
    resulting in 1,400 for training, 100 for validation and 100 for test
    """
    source_folder = "raw/RIMES"
    target_folder = "formatted/RIMES_paragraph"
    img_folder_path = os.path.join(target_folder, "images_gray")
    os.makedirs(target_folder, exist_ok=True)

    eval_xml_path = os.path.join(source_folder, "eval_2011_annotated.xml")
    train_xml_path = os.path.join(source_folder, "training_2011.xml")
    for label_path in [eval_xml_path, train_xml_path]:
        if not os.path.isfile(label_path):
            print("error - {} not found".format(label_path))
            exit(-1)

    tar_filenames = ["eval_2011_gray.tar", "training_2011_gray.tar"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    nb_training_samples = sum([1 if "train" in name else 0 for name in os.listdir(img_folder_path)])
    nb_valid_samples = 100
    begin_valid_ind = nb_training_samples - nb_valid_samples

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    for set_name in gt.keys():
        os.makedirs(os.path.join(target_folder, set_name), exist_ok=True)

    charset = set()
    for set_name, xml_path in zip(["train", "eval"], [train_xml_path, eval_xml_path]):
        xml_root = ET.parse(xml_path).getroot()
        for page in xml_root:
            name = page.attrib.get("FileName").split("/")[-1].split(".")[0]
            img_path = os.path.join(img_folder_path, name+".png")
            if set_name == "train":
                new_set_name = "train" if int(name.split("-")[-1])<begin_valid_ind else "valid"
            else:
                new_set_name = "test"
            fold = os.path.join(target_folder, new_set_name)
            new_name = "{}_{}.png".format(new_set_name, len(os.listdir(fold)))
            new_path = os.path.join(fold, new_name)
            os.replace(img_path, new_path)
            lines = []
            full_text = ""
            for i, line in enumerate(page[0]):

                lines.append({
                    "text": line.attrib.get("Value"),
                    "left": int(line.attrib.get("Left")),
                    "bottom": int(line.attrib.get("Bottom")),
                    "right": int(line.attrib.get("Right")),
                    "top": int(line.attrib.get("Top")),
                })
                full_text = "{}{}\n".format(full_text, lines[-1]["text"])
            paragraph = {
                "text": full_text[:-1],
                "lines": lines
            }
            charset = charset.union(set(full_text))
            gt[new_set_name][new_name] = paragraph

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
         pickle.dump({
             "ground_truth": gt,
             "charset": sorted(list(charset)),
         }, f)

    shutil.rmtree(img_folder_path)


def format_READ2016_line():
    """
    Format the READ 2016 dataset at line level with the official split (8,349 for training, 1,040 for validation and 1,138 for test)
    """
    source_folder = "raw/READ_2016"
    target_folder = "formatted/READ_2016_lines"
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    tar_filenames = ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    os.rename(os.path.join(target_folder, "PublicData", "Training"), os.path.join(target_folder, "train"))
    os.rename(os.path.join(target_folder, "PublicData", "Validation"), os.path.join(target_folder, "valid"))
    os.rename(os.path.join(target_folder, "Test-ICFHR-2016"), os.path.join(target_folder, "test"))
    os.rmdir(os.path.join(target_folder, "PublicData"))
    for set_name in ["train", "valid", ]:
        for filename in os.listdir(os.path.join(target_folder, set_name, "Images")):
            filepath = os.path.join(target_folder, set_name, "Images", filename)
            if os.path.isfile(filepath):
                os.rename(filepath, os.path.join(target_folder, set_name, filename))
        os.rmdir(os.path.join(target_folder, set_name, "Images"))

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    charset = set()
    for set_name in ["train", "valid", "test"]:
        img_fold_path = os.path.join(target_folder, set_name)
        xml_fold_path = os.path.join(target_folder, set_name, "page")
        i = 0
        for xml_file_name in sorted(os.listdir(xml_fold_path)):
            if xml_file_name.split(".")[-1] != "xml":
                continue
            filename = xml_file_name.split(".")[0]
            img_path = os.path.join(img_fold_path, filename+".JPG")
            xml_file_path = os.path.join(xml_fold_path, xml_file_name)
            xml_root = ET.parse(xml_file_path).getroot()
            img = np.array(Image.open(img_path))
            for text_region in xml_root[1][1:]:
                if text_region.tag.split("}")[-1] != "TextRegion":
                    continue
                for balise in text_region:
                    if balise.tag.split("}")[-1] != "TextLine":
                        continue
                    for sub in balise:
                        if sub.tag.split("}")[-1] == "Coords":
                            points = sub.attrib["points"].split(" ")
                            x_points, y_points = list(), list()
                            for p in points:
                                y_points.append(int(p.split(",")[1]))
                                x_points.append(int(p.split(",")[0]))
                        elif sub.tag.split("}")[-1] == "TextEquiv":
                            line_label = sub[0].text
                    if line_label is None:
                        continue
                    top, bottom, left, right = np.min(y_points), np.max(y_points), np.min(x_points), np.max(x_points)
                    new_img_name = "{}_{}.jpeg".format(set_name, i)
                    new_img_path = os.path.join(img_fold_path, new_img_name)
                    curr_img = img[top:bottom + 1, left:right + 1]
                    Image.fromarray(curr_img).save(new_img_path)
                    gt[set_name][new_img_name] = {"text": line_label, }
                    charset = charset.union(line_label)
                    i += 1
                    line_label = None
            os.remove(img_path)
        shutil.rmtree(xml_fold_path)

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)


def format_READ2016_paragraph():
    """
    Format the READ 2016 dataset at paragraph level with the official split (1,584 for training 179, for validation and 197 for test)
    """
    source_folder = "raw/READ_2016"
    target_folder = "formatted/READ_2016_paragraph"
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    tar_filenames = ["Test-ICFHR-2016.tgz", "Train-And-Val-ICFHR-2016.tgz"]
    tar_paths = [os.path.join(source_folder, name) for name in tar_filenames]
    for tar_path in tar_paths:
        if not os.path.isfile(tar_path):
            print("error - {} not found".format(tar_path))
            exit(-1)
        tar = tarfile.open(tar_path)
        tar.extractall(target_folder)
        tar.close()

    os.rename(os.path.join(target_folder, "PublicData", "Training"), os.path.join(target_folder, "train"))
    os.rename(os.path.join(target_folder, "PublicData", "Validation"), os.path.join(target_folder, "valid"))
    os.rename(os.path.join(target_folder, "Test-ICFHR-2016"), os.path.join(target_folder, "test"))
    os.rmdir(os.path.join(target_folder, "PublicData"))
    for set_name in ["train", "valid", ]:
        for filename in os.listdir(os.path.join(target_folder, set_name, "Images")):
            filepath = os.path.join(target_folder, set_name, "Images", filename)
            if os.path.isfile(filepath):
                os.rename(filepath, os.path.join(target_folder, set_name, filename))
        os.rmdir(os.path.join(target_folder, set_name, "Images"))

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }

    charset = set()
    for set_name in ["train", "valid", "test"]:
        img_fold_path = os.path.join(target_folder, set_name)
        xml_fold_path = os.path.join(target_folder, set_name, "page")
        i = 0
        for xml_file_name in os.listdir(xml_fold_path):
            if xml_file_name.split(".")[-1] != "xml":
                continue
            filename = xml_file_name.split(".")[0]
            img_path = os.path.join(img_fold_path, filename+".JPG")
            xml_file_path = os.path.join(xml_fold_path, xml_file_name)
            xml_root = ET.parse(xml_file_path).getroot()
            img = np.array(Image.open(img_path))
            for text_region in xml_root[1][1:]:
                if text_region.tag.split("}")[-1] != "TextRegion":
                    continue
                for balise in text_region:
                    if balise.tag.split("}")[-1] == "Coords":
                        points = balise.attrib["points"].split(" ")
                        x_points, y_points = list(), list()
                        for p in points:
                            y_points.append(int(p.split(",")[1]))
                            x_points.append(int(p.split(",")[0]))
                    if balise.tag.split("}")[-1] == "TextEquiv":
                        pg_label = balise[0].text
                if pg_label is None:
                    continue
                top, bottom, left, right = np.min(y_points), np.max(y_points), np.min(x_points), np.max(x_points)
                new_img_name = "{}_{}.jpeg".format(set_name, i)
                new_img_path = os.path.join(img_fold_path, new_img_name)
                curr_img = img[top:bottom + 1, left:right + 1]
                Image.fromarray(curr_img).save(new_img_path)
                gt[set_name][new_img_name] = {"text": pg_label, }
                charset = charset.union(pg_label)
                i += 1
                pg_label = None
            os.remove(img_path)
        shutil.rmtree(xml_fold_path)

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)


def format_SK_line(
    train_names: List[str],
    val_names: List[str],
    test_names: List[str],
    pages_folder: Union[Path, str],
    target_folder: Union[Path, str] = "formatted/SK_lines",
    skip_lines_with_grid: bool = True,
    img_format: str = "jpg",
    use_bw: bool = False
):
    """
    Format the Sukhovo-Kobylin dataset at line level
    """
    assert len(train_names) == len(set(train_names))
    assert len(val_names) == len(set(val_names))
    assert len(test_names) == len(set(test_names))
    assert len(set(train_names).intersection(val_names)) == 0
    assert len(set(train_names).intersection(test_names)) == 0
    assert len(set(val_names).intersection(test_names)) == 0

    pages_folder = Path(pages_folder)
    target_folder = Path(target_folder)
    assert Path.is_dir(pages_folder)
    if os.path.isdir(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)

    page_names = sorted(os.listdir(pages_folder))
    page_names = list(filter(
        lambda page_name: Path.is_dir(pages_folder / page_name),
        page_names
    ))

    pages_in_params_set = set(train_names).union(set(val_names)).union(set(test_names))
    if set(page_names) != pages_in_params_set:
        unused = set(page_names).difference(pages_in_params_set)
        logger.warning("Unused pages: %s", str(unused))

    for set_pages in [train_names, val_names, test_names]:
        pages_not_exist = set(set_pages).difference(set(page_names))
        if len(pages_not_exist):
            logger.warning("Pages not exist: %s", str(pages_not_exist))

    train_names = list(filter(lambda page: page in page_names, train_names))
    val_names = list(filter(lambda page: page in page_names, val_names))
    test_names = list(filter(lambda page: page in page_names, test_names))

    def get_lines_v2(
        page_name: str,
        pages_path: Union[Path, str],
        img_format: str = img_format
    ) -> Optional[tuple[list[np.ndarray], list[str]]]:
        #page_path = Path(pages_path) / page_name
        txt_path = Path(pages_path) / page_name
        img_path = txt_path / "BW" if use_bw else txt_path
        assert txt_path.is_dir()
        assert img_path.is_dir()
        line_imgs = []
        text_lines = []

        def key_func(filename: str):
            """Key-функция для сортировки директории страницы."""
            pos_1 = filename.find("_")
            pos_2 = filename.find(".")
            try:
                return int(filename[pos_1 + 1:pos_2])
            except:
                return -1

        txt_list_dir = sorted(os.listdir(txt_path), key=key_func)
        img_list_dir = sorted(os.listdir(img_path), key=key_func)
        for file_name in txt_list_dir:
            point_pos = file_name.find(".")
            file_format = file_name[point_pos + 1:]
            file_name = file_name[:point_pos]
            if file_format != "txt":
                continue
            if "#" in file_name:
                continue
            img_full_name = f"{file_name}bw.{img_format}" if use_bw else f"{file_name}.{img_format}"
            if img_full_name not in img_list_dir:
                logger.warning(f"Image for %s.txt does not exists", file_name)
                return None
            with open(txt_path / f"{file_name}.txt", encoding="utf-8-sig") as line_file:
                text_line = line_file.readlines()[0]
                text_line = text_line.replace("\n", "").replace("$", "").replace("\u200b", "").strip()
                if text_line == "":
                    logger.warning(f"Empty text in %s.txt", file_name)
                    continue
                if text_line == "Экспертная расшифровка отсутствует":
                    logger.warning(f"Экспертная расшифровка отсутствует в %s.txt", file_name)
                    return None
                if text_line == "Авторасшифровка отсутствует":
                    logger.warning(f"Авторасшифровка отсутствует в %s.txt", file_name)
                    return None
            img = plt.imread(img_path / img_full_name)
            if img.shape[0] < 10 or img.shape[1] < 10:
                logger.warning(f"img.shape[0] < 10 or img.shape[1] < 10 for %s (shape: %s)", img_full_name, str(img.shape))
                continue
            line_imgs.append(img)
            text_lines.append(text_line)

        return line_imgs, text_lines

    gt = {
        "train": dict(),
        "valid": dict(),
        "test": dict()
    }
    charset = set()

    for page_names, set_name in zip(
        [train_names, val_names, test_names],
        ["train", "valid", "test"]
    ):
        i = 0
        num_strings_with_hashtag = 0
        img_fold_path = os.path.join(target_folder, set_name)
        os.makedirs(img_fold_path, exist_ok=True)
        for name in page_names:
            r = get_lines_v2(name, pages_folder)
            if r is None:
                logger.warning("Error in name: %s, set: %s", name, set_name)
                continue
            line_imgs, text_lines = r
            for line_img, text in zip(line_imgs, text_lines):
                if skip_lines_with_grid and "#" in text:
                    logger.debug("Skipping %s because '#' in name, set: %s, text: %s",
                                 name, set_name, text)
                    num_strings_with_hashtag += 1
                    continue
                new_img_name = f"{set_name}_{i}.jpg"
                new_img_path = os.path.join(img_fold_path, new_img_name)
                try:
                    Image.fromarray(line_img).save(new_img_path)
                except:
                    logger.warning("Error while saving %s, set: %s, text: %s",
                                   name, set_name, text)
                    continue
                gt[set_name][new_img_name] = {"text": text, }
                charset = charset.union(text)
                i += 1
        logger.info("Number of string with '#' in %s: %d (не учитывем # в названии txt-файлов)",
                    set_name, num_strings_with_hashtag)

    with open(os.path.join(target_folder, "labels.pkl"), "wb") as f:
        pickle.dump({
            "ground_truth": gt,
            "charset": sorted(list(charset)),
        }, f)

    logger.info("Train names: %s", str(train_names))
    logger.info("Val names: %s", str(val_names))
    logger.info("Test names: %s", str(test_names))
    


if __name__ == "__main__":

    # format_IAM_line()
    # format_IAM_paragraph()

    # format_RIMES_line()
    # format_RIMES_paragraph()

    # format_READ2016_line()
    # format_READ2016_paragraph()

    train_names = ['4', '4об', '5', '5об', '6', '6об', '7', '8', '8об', '9', '9об', '10','10об', '11', '14', '14об',
                   '15', '15об', '16', '16об', '17', '17об', '18', '18об', '19', '19об', '20', '20об', '21', '21об',
                   '22', '22об', '23', '23об', '24', '24об', '25', '25об', '26', '26об', '27', '27об', '28', '28об',
                   '29', '29об', '2об', '30', '30об', '31', '31об', '32', '32об', '33', '33об', '34', '34об', '35',
                   '35об', '36', '38', '38об', '39', '39об', '40', '40об', '41', '41об', '42', '42об', '43', '43об',
                   '44', '44об', '45об', '46', '46об', '47', '47об', '48', '48об', '49', '49об', '50', '50об', '51',
                   '51об', '52', '52об', '53', '53об', '54', '54об', '55', '55об', '56', '56об', '57', '57об', '58',
                   '67', '67об', '68', '68об', '69', '69об',  '70', '70об', '71', '71об', '72', '72об', '73', '73об',
                   '74', '76', '83об', '85', '88об', '89', '66об', '84', '84об', '90']
    val_names = ['58об', '59', '59об', '60', '60об']
    test_names = ['61', '75', '75об', '7об', '95']

    format_SK_line(
        train_names=train_names,
        val_names=val_names,
        test_names=test_names,
        pages_folder="raw/SK/Pages_v7",
        img_format="bmp"
    )
