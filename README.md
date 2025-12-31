# Dataset Formater

## Overview

`dataset_formater` is designed to help you:

* Load datasets in **YOLO**, **COCO**, or **COCO-JSON** format into a common `DatasetIR` interface.
* Manipulate images, annotations and categories programmatically (remap classes, subset, filter, etc.).
* Export back to standard formats so the processed dataset is ready for training.
* Generate **automatic dataset reports** (PDF) with statistics and plots.
* Convert **bounding-box-only datasets** into **instance segmentation datasets** using **Segment Anything (SAM)**.

The goal is to centralize typical dataset housekeeping tasks in a single, consistent toolkit.

---

## 3. Installation

> To be completed (pip install / editable install instructions).

---

## Scripts

### `transform_dataset.py`

**Description**  
Convert a dataset between `yolo`, `coco` and `coco_json` formats, preserving the `train` / `val` / `test` split structure.

**Arguments**

* `--source_path` (required): Path to the source dataset root folder.
* `--source_format` (required): Source format: `yolo`, `coco` or `coco_json`.
* `--dest_path` (required): Path where the converted dataset will be saved.
* `--dest_format` (required): Output format: `yolo`, `coco` or `coco_json`.

**Example**

```bash
python -m dataset_formater.scripts.transform_dataset \
  --source_path ./dataset_formater/dataset_yolo11 \
  --source_format yolo \
  --dest_path ./dataset_formater/dataset_coco_json \
  --dest_format coco_json
