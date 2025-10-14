import os
import pandas as pd
import shutil
import random

# Set paths
original_data_dir = r"C:\Users\BIGNA\Desktop\ham10000\archive" # Will need to change for your location if running yourself
sample_data_dir = r"sample_dataset"
sample_images_dir = os.path.join(sample_data_dir, "images")

# Ensure sample images directory exists
os.makedirs(sample_images_dir, exist_ok=True)

# Load metadata
metadata = pd.read_csv(os.path.join(original_data_dir, "HAM10000_metadata.csv"))

# Define lesion types
lesion_types = metadata['dx'].unique()

# Create sample dataset with 5 images per class
samples_per_class = 20
sample_image_ids = []

for lesion_type in lesion_types:
    # Get all image IDs for this lesion type
    class_image_ids = metadata[metadata['dx'] == lesion_type]['image_id'].values
    # Randomly select samples_per_class images
    selected_ids = random.sample(list(class_image_ids), min(samples_per_class, len(class_image_ids)))
    sample_image_ids.extend(selected_ids)

# Create a filtered metadata file with only the selected images
sample_metadata = metadata[metadata['image_id'].isin(sample_image_ids)]
sample_metadata.to_csv(os.path.join(sample_data_dir, "HAM10000_metadata.csv"), index=False)

# Copy selected images
copied_count = 0
for image_id in sample_image_ids:
    # Search for the image in both part folders
    found = False
    for part_folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        source_path = os.path.join(original_data_dir, part_folder, f"{image_id}.jpg")
        if os.path.exists(source_path):
            dest_path = os.path.join(sample_images_dir, f"{image_id}.jpg")
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            found = True
            break

    if not found:
        print(f"Could not find image: {image_id}")

print(f"Created sample dataset with {copied_count} images")
print(f"Sample metadata saved to {os.path.join(sample_data_dir, 'HAM10000_metadata.csv')}")