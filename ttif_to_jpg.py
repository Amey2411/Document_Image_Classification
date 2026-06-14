from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT = Path(r"D:\assignment_dataset")

IMAGE_ROOT = ROOT / "images"
LABEL_ROOT = ROOT / "labels"

OUTPUT_ROOT = Path(r"D:\assignment_dataset\df")

CLASS_NAMES = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific_report",
    6: "scientific_publication",
    7: "specification",
    8: "file_folder",
    9: "news_article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo"
}

for split in ["train", "valid", "test"]:
    for cls in CLASS_NAMES.values():
        (OUTPUT_ROOT / split / cls).mkdir(
            parents=True,
            exist_ok=True
        )

def process_split(label_file, split_name):

    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=split_name):

        rel_path, label = line.strip().split()

        label = int(label)

        src = IMAGE_ROOT / rel_path

        class_name = CLASS_NAMES[label]

        jpg_name = src.stem + ".jpg"

        dst = (
            OUTPUT_ROOT /
            split_name /
            class_name /
            jpg_name
        )

        try:
            img = Image.open(src).convert("RGB")

            img.save(
                dst,
                "JPEG",
                quality=95,
                optimize=True
            )

        except Exception as e:
            print(f"Error: {src}")
            print(e)

process_split(
    LABEL_ROOT / "train.txt",
    "train"
)

process_split(
    LABEL_ROOT / "val.txt",
    "valid"
)

process_split(
    LABEL_ROOT / "test.txt",
    "test"
)

print("Done")