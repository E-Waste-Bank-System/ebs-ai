import os
import json
from icrawler.builtin import BingImageCrawler

# Load needed images per class
with open('needed_images_to_balance.json', 'r') as f:
    needed = json.load(f)

# Where to save new images
dest_root = 'images_to_add'

# For each class, download the needed number of images
for cls, num_needed in needed.items():
    dest_dir = os.path.join(dest_root, cls.replace('/', '_'))
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Downloading {num_needed} images for class '{cls}'...")
    crawler = BingImageCrawler(storage={'root_dir': dest_dir})
    crawler.crawl(keyword=cls, max_num=num_needed)

print("Done downloading images for all underrepresented classes.") 