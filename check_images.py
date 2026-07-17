import os
from PIL import Image

dataset_path = "Dataset"

for category in ["with_mask", "without_mask"]:
    folder = os.path.join(dataset_path, category)
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception as e:
            print(f"BAD FILE: {filepath} -> {e}")

print("Check complete.")