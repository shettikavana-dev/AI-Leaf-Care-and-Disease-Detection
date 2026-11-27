from pathlib import Path

# Main dataset directory
dataset_path = Path("dataset")

# Supported image formats
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

plants = ["tomato", "potato", "mango"]

print("\n🌿 AI LEAF CARE - DATASET CHECK\n")

for plant in plants:
    plant_path = dataset_path / plant

    if not plant_path.exists():
        print(f"{plant.upper()}: Folder not found")
        continue

    images = [
        file for file in plant_path.rglob("*")
        if file.is_file()
        and file.suffix.lower() in image_extensions
    ]

    print(f"{plant.upper()}: {len(images)} images found")

print("\nDataset checking completed!")