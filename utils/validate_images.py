from pathlib import Path
from PIL import Image, UnidentifiedImageError

dataset_path = Path("dataset")
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

print("\n🌿 AI LEAF CARE - IMAGE VALIDATION\n")

for plant in ["tomato", "potato", "mango"]:
    plant_path = dataset_path / plant

    valid = 0
    corrupted = 0

    for file in plant_path.rglob("*"):
        if file.is_file() and file.suffix.lower() in image_extensions:
            try:
                with Image.open(file) as img:
                    img.verify()
                valid += 1

            except (UnidentifiedImageError, OSError, ValueError):
                corrupted += 1

    print(f"\n{plant.upper()}")
    print(f"Valid images: {valid}")
    print(f"Corrupted images: {corrupted}")

print("\nImage validation completed!")

