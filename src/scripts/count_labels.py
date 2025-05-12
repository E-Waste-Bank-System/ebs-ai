import csv
import json
from collections import Counter
from src.utils.label_utils import count_rectangle_labels

def print_label_statistics(label_counter, total_annotations):
    """
    Print statistics about the labels
    """
    print(f"Total annotations: {total_annotations}")
    print("\nLabel counts:")
    print("-" * 40)
    
    # Sort labels by count in descending order
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate the maximum label length for alignment
    max_label_length = max(len(label) for label, _ in sorted_labels) if sorted_labels else 0
    
    # Print each label with its count and percentage
    for label, count in sorted_labels:
        percentage = (count / total_annotations) * 100
        print(f"{label:{max_label_length}} | {count:5} | {percentage:6.2f}%")

def main():
    csv_file = "src/data/annotations/dataset.csv"
    
    print("Counting rectangle labels in the dataset...")
    label_counter, total_annotations = count_rectangle_labels(csv_file)
    
    print("\nResults:")
    print_label_statistics(label_counter, total_annotations)
    
    # Print suggestions for balancing the dataset
    print("\nSuggestions for balancing the dataset:")
    print("-" * 40)
    
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[1])
    
    if sorted_labels:
        median_count = sorted_labels[len(sorted_labels) // 2][1]
        max_count = max(count for _, count in sorted_labels)
        
        for label, count in sorted_labels:
            if count < median_count:
                print(f"Add more '{label}' images (currently {count}, median is {median_count})")
            elif count < max_count * 0.5:
                print(f"Consider adding more '{label}' images (currently {count}, less than 50% of max class)")

if __name__ == "__main__":
    main() 