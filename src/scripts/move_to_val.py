import os
import shutil
import random
from collections import defaultdict
from src.utils.label_utils import get_image_classes

def move_images_to_val():
    """
    Move one image from each class from train to val directory, along with their label files
    """
    image_classes = get_image_classes('src/data/annotations/dataset.csv')
    
    # Create a mapping of classes to images
    class_to_images = defaultdict(list)
    for image, classes in image_classes.items():
        for cls in classes:
            class_to_images[cls].append(image)
    
    # Create validation directories if they don't exist
    os.makedirs('src/data/images/val', exist_ok=True)
    os.makedirs('src/data/labels/val', exist_ok=True)
    
    # Track which images we've moved
    moved_images = set()
    
    # For each class, move one image to validation
    for cls, images in class_to_images.items():
        # Filter out images that have already been moved
        available_images = [img for img in images if img not in moved_images]
        
        if available_images:
            # Randomly select one image
            selected_image = random.choice(available_images)
            
            # Source and destination paths for image
            img_src_path = os.path.join('src/data/images/train', selected_image)
            img_dst_path = os.path.join('src/data/images/val', selected_image)
            
            # Source and destination paths for label file
            # Label files typically have the same name but with .txt extension
            label_filename = os.path.splitext(selected_image)[0] + '.txt'
            label_src_path = os.path.join('src/data/labels/train', label_filename)
            label_dst_path = os.path.join('src/data/labels/val', label_filename)
            
            # Only move if both the image and label file exist in the train directory
            if os.path.exists(img_src_path) and os.path.exists(label_src_path):
                print(f"Moving {selected_image} (class: {cls}) from train to val")
                # Copy the image file
                shutil.copy2(img_src_path, img_dst_path)  # Use copy2 to preserve metadata
                
                # Copy the label file
                print(f"Moving label file {label_filename} from train to val")
                shutil.copy2(label_src_path, label_dst_path)
                
                moved_images.add(selected_image)
            elif os.path.exists(img_src_path) and not os.path.exists(label_src_path):
                print(f"Warning: Label file {label_filename} not found for image {selected_image}")
                # Still copy the image if no label file is found
                shutil.copy2(img_src_path, img_dst_path)
                moved_images.add(selected_image)
            else:
                print(f"Warning: Image {selected_image} not found in train directory")
    
    # Print summary
    print(f"\nTotal classes: {len(class_to_images)}")
    print(f"Total images moved to validation: {len(moved_images)}")
    print(f"Classes represented in validation: {len(moved_images)}")

if __name__ == "__main__":
    move_images_to_val() 