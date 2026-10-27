
from pathlib import Path
import random
import shutil

# Original dataset folder
source = Path(r"C:\Users\ASUS\Downloads\Garbage data\garbage-dataset")

# Project dataset folder
project = Path(__file__).parent
dataset = project / "Dataset"

categories = ["plastic", "paper", "cardboard"]
random.seed(42)
extensions = {".jpg", ".jpeg", ".png"}

if not source.is_dir():
    raise FileNotFoundError(f"Original dataset not found: {source}")

for category in categories:
    source_folder = source / category

    if not source_folder.is_dir():
        raise FileNotFoundError(f"Category folder not found: {source_folder}")

    images = [
        file for file in source_folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    ]

    if not images:
        raise ValueError(f"No images found in {source_folder}")

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:validation_end],
        "test": images[validation_end:],
    }

    # Remove old images from these project folders
    for split in splits:
        destination = dataset / split / category
        destination.mkdir(parents=True, exist_ok=True)

        for old_file in destination.iterdir():
            if old_file.is_file() and old_file.suffix.lower() in extensions:
                old_file.unlink()

    # Copy images into the correct folders
    for split, files in splits.items():
        destination = dataset / split / category

        for image in files:
            shutil.copy2(image, destination / image.name)

    print(
        f"{category}: {len(splits['train'])} train, "
        f"{len(splits['validation'])} validation, "
        f"{len(splits['test'])} test images"
    )

print("\nDataset preparation complete!")
