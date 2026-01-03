import argparse
from visionDS_toolkit.utilities.dataset_folder_interface import DatasetFolder
from visionDS_toolkit.utilities.dump_functions import (
    select_dump_function,
    dump_yolo_dataset,
    dump_coco_estandar_dataset,
    dump_coco_json_dataset,
)

# =========================
# DEFAULT CONSTANTS
# =========================
SOURCE_PATH = ""  # p.ej. "data/source_yolo"
SOURCE_FORMAT = ""  # "yolo" | "coco" | "coco_json"
DEST_PATH = ""  # p.ej. "data/dest_coco"
DEST_FORMAT = ""  # "yolo" | "coco" | "coco_json"


# FORMATS SUPPORTED: coco, coco_json, yolo
def transform_dataset(
    source_path: str,
    source_format: str,
    dest_path: str,
    dest_format: str,
):
    print(f"Init transformation from {source_format} to {dest_format}")

    dataset = DatasetFolder(path=source_path, dataset_type=source_format)
    dumper = select_dump_function(fmt=dest_format)
    dumper(dataset, dest_path)
    print(f"Transformation completed. Dataset saved in {dest_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Transform dataset between different formats (YOLO, COCO, COCO JSON)."
    )

    parser.add_argument(
        "--source_path",
        type=str,
        default=SOURCE_PATH,  
        help="Path to the source dataset folder.",
    )
    parser.add_argument(
        "--source_format",
        type=str,
        default=SOURCE_FORMAT,  
        choices=["yolo", "coco", "coco_json"],
        help="Format of the source dataset.",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        default=DEST_PATH, 
        help="Path to save the transformed dataset.",
    )
    parser.add_argument(
        "--dest_format",
        type=str,
        default=DEST_FORMAT,  
        choices=["yolo", "coco", "coco_json"],
        help="Format of the destination dataset.",
    )

    args = parser.parse_args()

    if not args.source_path:
        raise ValueError(
            "source_path is empty. Set SOURCE_PATH en el script o pasa --source_path."
        )
    if not args.dest_path:
        raise ValueError(
            "dest_path is empty. Set DEST_PATH en el script o pasa --dest_path."
        )
    if not args.source_format:
        raise ValueError(
            "source_format is empty. Set SOURCE_FORMAT en el script o pasa --source_format (yolo|coco|coco_json)."
        )
    if not args.dest_format:
        raise ValueError(
            "dest_format is empty. Set DEST_FORMAT en el script o pasa --dest_format (yolo|coco|coco_json)."
        )

    transform_dataset(
        source_path=args.source_path,
        source_format=args.source_format,
        dest_path=args.dest_path,
        dest_format=args.dest_format,
    )
if __name__ == "__main__":
    main()