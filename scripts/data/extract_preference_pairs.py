"""
Extract preference pairs (+, -) from VisualPRM400K annotation JSONL files.
Each pair shares the same image; one sample is labeled '+' and the other '-'.
Images are copied to output dir and a single swiftdata.jsonl is written.
Configure paths via environment variables (see scripts/config.example.env).
"""

import os
import json
import shutil
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Path configuration (override via env)
# -----------------------------------------------------------------------------
VISUAL_PRM_ROOT = os.environ.get("VISUAL_PRM_ROOT", "/path/to/VisualPRM400K")
IMAGE_SUBSET = os.environ.get("IMAGE_SUBSET", "GeomVerse")
OUTPUT_BASE_DIR = os.environ.get("OUTPUT_BASE_DIR", "/path/to/evpv_data")

# Derived paths
SOURCE_IMAGE_BASE_DIR = os.path.join(VISUAL_PRM_ROOT, "images", IMAGE_SUBSET)
SOURCE_ANNOTATION_DIR = os.path.join(VISUAL_PRM_ROOT, "annotations")

# JSONL files to process (under SOURCE_ANNOTATION_DIR)
JSONL_FILENAMES = [
    "geomverse_extracted_pairs_vqa_correctness_rules.jsonl",
    "geomverse_extracted_prm_process.jsonl",
    "geomverse_extracted_pairs_vqa_format_rules.jsonl",
]

OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_BASE_DIR, IMAGE_SUBSET, "image")
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_BASE_DIR, IMAGE_SUBSET, "swiftdata.jsonl")


def main():
    print(f"Creating output image dir: {OUTPUT_IMAGE_DIR}")
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    processed_images = set()
    final_data_pairs = []

    for filename in tqdm(JSONL_FILENAMES, desc="Processing JSONL files"):
        file_path = os.path.join(SOURCE_ANNOTATION_DIR, filename)

        if not os.path.exists(file_path):
            print(f"Warning: file not found, skipping - {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            i = 0
            pbar = tqdm(total=len(lines), desc=f"  Scanning {filename}", leave=False)
            while i < len(lines) - 1:
                try:
                    data1 = json.loads(lines[i])
                    data2 = json.loads(lines[i + 1])

                    image_path1 = data1.get("image")
                    image_path2 = data2.get("image")

                    if image_path1 and image_path1 == image_path2:
                        if image_path1 not in processed_images:
                            rating1 = data1["conversations"][-1]["value"]
                            rating2 = data2["conversations"][-1]["value"]

                            if {rating1, rating2} == {"+", "-"}:
                                source_img_full = os.path.join(
                                    SOURCE_IMAGE_BASE_DIR, image_path1
                                )
                                dest_img_full = os.path.join(
                                    OUTPUT_IMAGE_DIR, image_path1
                                )

                                if os.path.exists(source_img_full):
                                    os.makedirs(
                                        os.path.dirname(dest_img_full),
                                        exist_ok=True,
                                    )
                                    shutil.copy2(source_img_full, dest_img_full)
                                else:
                                    print(
                                        f"Warning: source image not found, skip copy: {source_img_full}"
                                    )

                                new_relative_path = os.path.join("image", image_path1)
                                data1["image"] = new_relative_path
                                data2["image"] = new_relative_path

                                final_data_pairs.append(data1)
                                final_data_pairs.append(data2)
                                processed_images.add(image_path1)

                                pbar.update(2)
                                i += 2
                                continue

                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

                pbar.update(1)
                i += 1
            pbar.close()

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    print(
        f"Done. Found {len(processed_images)} unique images, {len(final_data_pairs)} records."
    )
    print(f"Writing to: {OUTPUT_JSONL_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL_PATH), exist_ok=True)
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f_out:
        for item in tqdm(final_data_pairs, desc="Writing"):
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("All done.")


if __name__ == "__main__":
    main()
