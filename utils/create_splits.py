from pathlib import Path
import csv
import random

dataset_path = Path("dataset")
output_file = Path("data_splits.csv")

plants = ["tomato", "potato", "mango"]
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

if output_file.exists():
    raise SystemExit(
        "data_splits.csv already exists. Please check before overwriting it."
    )

random.seed(42)
rows = []

for plant in plants:
    plant_path = dataset_path / plant

    for class_folder in sorted(plant_path.iterdir()):
        if not class_folder.is_dir():
            continue

        images = [
            file for file in class_folder.rglob("*")
            if file.is_file() and file.suffix.lower() in extensions
        ]

        random.shuffle(images)

        total = len(images)
        train_end = int(total * 0.70)
        val_end = train_end + int(total * 0.15)

        for index, image_path in enumerate(images):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "validation"
            else:
                split = "test"

            rows.append({
                "filepath": image_path.as_posix(),
                "plant": plant,
                "class_name": class_folder.name,
                "label": f"{plant}__{class_folder.name}",
                "split": split
            })

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filepath", "plant", "class_name", "label", "split"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nDATASET SPLITTING COMPLETED!")
print(f"Total images: {len(rows)}")

for split in ["train", "validation", "test"]:
    count = sum(1 for row in rows if row["split"] == split)
    print(f"{split.capitalize()}: {count}")

print(f"\nManifest saved as: {output_file}")