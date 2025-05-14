import csv
import json
import os
import shutil
from collections import Counter
from src.utils.label_utils import get_image_classes

# Step 1: Count images per class
image_classes = get_image_classes('src/data/annotations/dataset.csv')

# Step 2: Count images per class
class_counts = {cls: len(imgs) for cls, imgs in image_classes.items()}

# Step 3: Determine the target (max) count
if class_counts:
    target_count = max(class_counts.values())
else:
    target_count = 0

# Step 4: List underrepresented classes and how many images are needed
needed = {}
for cls, count in class_counts.items():
    if count < target_count:
        needed[cls] = target_count - count

# Step 5: Save to JSON
with open('needed_images_to_balance.json', 'w') as f:
    json.dump(needed, f, indent=2)

print(f"Saved needed images per class to needed_images_to_balance.json")

# Step 6: Copy images of underrepresented classes to images_to_add/CLASS_NAME/
source_dir = 'src/data/images/train'
dest_root = 'images_to_add'
os.makedirs(dest_root, exist_ok=True)

for cls in needed:
    dest_dir = os.path.join(dest_root, cls.replace('/', '_'))
    os.makedirs(dest_dir, exist_ok=True)
    for img_file in image_classes[cls]:
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(dest_dir, img_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
print(f"Copied images for underrepresented classes to {dest_root}/CLASS_NAME/") 