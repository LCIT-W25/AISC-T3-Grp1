import tarfile
import os
import matplotlib.pyplot as plt

# Path to your tar file and extraction directory
tar_file_path = r'C:\Users\kiran\Downloads\Yelp Photos\yelp_photos.tar'  # Replace with the actual path to your tar file
extraction_dir = r'C:\Users\kiran\Downloads\Yelp Photos\images'    # Replace with the desired extraction path

# Create the extraction directory if it doesn't exist
if not os.path.exists(extraction_dir):
    os.makedirs(extraction_dir)

# Open the tar file and extract all contents
with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall(path=extraction_dir)

# List the extracted files (image files)
extracted_files = os.listdir(extraction_dir)
print(extracted_files[:5])  # Display the first 5 extracted image filenames
