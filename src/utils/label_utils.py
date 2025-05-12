import os
import csv
import json
from collections import defaultdict, Counter

def get_image_classes(csv_path):
    """
    Read the dataset.csv file and create a mapping of image filenames to their classes
    """
    image_classes = defaultdict(list)
    with open(csv_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            if row['label']:
                try:
                    image_path = row['image']
                    image_filename = os.path.basename(image_path)
                    label_data = json.loads(row['label'])
                    for annotation in label_data:
                        if 'rectanglelabels' in annotation:
                            for label in annotation['rectanglelabels']:
                                image_classes[image_filename].append(label)
                except json.JSONDecodeError:
                    print(f"Error parsing JSON in row: {row.get('id', '?')}")
                    continue
    return image_classes

def count_rectangle_labels(csv_file):
    """
    Count occurrences of each rectanglelabels value in the dataset.csv file
    """
    label_counter = Counter()
    total_annotations = 0
    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            if row['label']:
                try:
                    label_data = json.loads(row['label'])
                    for annotation in label_data:
                        if 'rectanglelabels' in annotation:
                            for label in annotation['rectanglelabels']:
                                label_counter[label] += 1
                                total_annotations += 1
                except json.JSONDecodeError:
                    print(f"Error parsing JSON in row: {row.get('id', '?')}")
                    continue
    return label_counter, total_annotations 