import pandas as pd
import os

def add_image_path_column(csv_path, split_name, image_extension=".tif"):
    """
    Adds an 'image_path' column to the CSV file based on category, location_id, and image_id.

    Args:
        csv_path (str): Path to the CSV file.
        split_name (str): One of 'train', 'val', 'test_gt'.
        image_extension (str): File extension of the images (default: .tiff).
    """
    df = pd.read_csv(csv_path)

    def build_path(row):
        category = row['category']
        location_id = row['location_id']
        image_id = row['image_id']

        prefix = "../Datasets/sentinel"
        folder = f"{category}_{location_id}"
        filename = f"{category}_{location_id}_{image_id}{image_extension}"
        return f"{prefix}/{split_name}/{category}/{folder}/{filename}"

    df['image_path'] = df.apply(build_path, axis=1)

    # Save back to same file (overwrite)
    df.to_csv(csv_path, index=False)
    print(f"Updated: {csv_path}")

if __name__ == "__main__":
    # List of CSV files and their corresponding split names
    files = [
        ("sentinel/train.csv", "train"),
        ("sentinel/val.csv", "val"),
        ("sentinel/test_gt.csv", "test_gt")
    ]

    for csv_file, split in files:
        if os.path.exists(csv_file):
            add_image_path_column(csv_file, split)
        else:
            print(f"File not found: {csv_file}")