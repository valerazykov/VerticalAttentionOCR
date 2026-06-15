#!/usr/bin/env python3
"""Format new Sukhovo-Kobylin line images for VAN inference.

The raw folders are expected to look like:

    raw/SK_new_data/
      Pages_438-1-224/
        14/
          14_1.bmp
          14_1.txt   # optional, often empty/BOM-only
          ...
      Pages_438-1-227/
        2/
          2_1.bmp
          ...

The output follows the format consumed by VerticalAttentionOCR.basic.OCRDataset:

    formatted/SK_new_data_lines/
      test/test_0.jpg
      test/test_1.jpg
      labels.pkl

All lines are put into the ``test`` split because this dataset is intended for
inference/visual inspection, not training.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


LOGGER = logging.getLogger(__name__)
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def clean_text(text: str) -> str:
    return (
        (text or "")
        .replace("\ufeff", "")
        .replace("\u200b", "")
        .replace("$", "")
        .strip()
    )


def load_charset(charset_source: Path | None) -> set[str]:
    if charset_source is None or not charset_source.exists():
        return set()
    with charset_source.open("rb") as f:
        info = pickle.load(f)
    return set(info.get("charset", []))


def page_sort_key(path: Path):
    name = path.name
    is_obverse = name.endswith("об")
    base = name[:-2] if is_obverse else name
    try:
        number = int(base)
    except ValueError:
        number = 10**9
    return (number, int(is_obverse), name)


def line_sort_key(path: Path):
    match = re.search(r"_(\d+)$", path.stem)
    number = int(match.group(1)) if match else 10**9
    return (number, path.name)


def discover_archives(raw_root: Path, archive_names: Iterable[str] | None) -> list[Path]:
    if archive_names:
        archives = [raw_root / name for name in archive_names]
    else:
        archives = [p for p in raw_root.iterdir() if p.is_dir()]
    missing = [p for p in archives if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing archive folders: {missing}")
    return sorted(archives, key=lambda p: p.name)


def read_optional_text(image_path: Path) -> str:
    txt_path = image_path.with_suffix(".txt")
    if not txt_path.exists():
        return ""
    try:
        return clean_text(txt_path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError:
        return clean_text(txt_path.read_text(encoding="cp1251", errors="ignore"))


def save_as_jpg(src: Path, dst: Path) -> None:
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        image.save(dst, quality=95)


def format_sk_new_data_lines(
    raw_root: str | Path = "raw/SK_new_data",
    target_folder: str | Path = "formatted/SK_new_data_lines",
    archive_names: Iterable[str] | None = ("Pages_438-1-224", "Pages_438-1-227"),
    charset_source: str | Path | None = "formatted/SK_lines/labels.pkl",
    add_txt_files: bool = True,
    overwrite: bool = True,
    dry_run: bool = False,
) -> dict:
    raw_root = Path(raw_root)
    target_folder = Path(target_folder)
    charset_source = Path(charset_source) if charset_source is not None else None

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    archives = discover_archives(raw_root, archive_names)
    items = []

    for archive in archives:
        pages = sorted([p for p in archive.iterdir() if p.is_dir()], key=page_sort_key)
        for page in pages:
            images = sorted(
                [p for p in page.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                key=line_sort_key,
            )
            for image_path in images:
                line_match = re.search(r"_(\d+)$", image_path.stem)
                line_id = int(line_match.group(1)) if line_match else None
                items.append({
                    "archive": archive.name,
                    "page": page.name,
                    "line_id": line_id,
                    "source_image": image_path,
                    "source_text": image_path.with_suffix(".txt"),
                    "text": read_optional_text(image_path),
                })

    LOGGER.info("Found %d line images", len(items))
    if dry_run:
        return {"n_items": len(items), "target_folder": str(target_folder)}

    if target_folder.exists() and overwrite:
        shutil.rmtree(target_folder)
    (target_folder / "train").mkdir(parents=True, exist_ok=True)
    (target_folder / "valid").mkdir(parents=True, exist_ok=True)
    (target_folder / "test").mkdir(parents=True, exist_ok=True)

    gt = {"train": {}, "valid": {}, "test": {}}
    metadata = {"test": {}}
    charset = load_charset(charset_source)

    for index, item in enumerate(items):
        new_name = f"test_{index}.jpg"
        dst = target_folder / "test" / new_name
        save_as_jpg(item["source_image"], dst)

        text = item["text"]
        if add_txt_files:
            dst.with_suffix(".txt").write_text(text, encoding="utf-8")

        gt["test"][new_name] = {"text": text}
        charset.update(text)
        metadata["test"][new_name] = {
            "archive": item["archive"],
            "page": item["page"],
            "line_id": item["line_id"],
            "source_image": str(item["source_image"]),
            "source_text": str(item["source_text"]),
            "raw_text": text,
        }

    labels = {
        "ground_truth": gt,
        "charset": sorted(charset),
        "metadata": metadata,
        "source": {
            "raw_root": str(raw_root),
            "archive_names": [p.name for p in archives],
            "charset_source": str(charset_source) if charset_source is not None else None,
        },
    }
    with (target_folder / "labels.pkl").open("wb") as f:
        pickle.dump(labels, f)

    LOGGER.info("Saved formatted dataset to %s", target_folder)
    return {"n_items": len(items), "target_folder": str(target_folder)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="raw/SK_new_data")
    parser.add_argument("--target-folder", default="formatted/SK_new_data_lines")
    parser.add_argument("--archive", dest="archives", action="append", default=None)
    parser.add_argument("--charset-source", default="formatted/SK_lines/labels.pkl")
    parser.add_argument("--no-txt-files", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    result = format_sk_new_data_lines(
        raw_root=args.raw_root,
        target_folder=args.target_folder,
        archive_names=args.archives,
        charset_source=args.charset_source,
        add_txt_files=not args.no_txt_files,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
    )
    LOGGER.info("Result: %s", result)


if __name__ == "__main__":
    main()
