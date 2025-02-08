# Donwload zip file from https://www.yelp.com/dataset/download
# Un-Zip it After donwloading

# Run the below code to get the review file with modified tar_filepath.


import tarfile

source_filepath = r'C:\Users\kiran\Downloads\Yelp JSON\yelp_dataset.tar'
destination_filepath = r'C:\Users\kiran\Downloads\Yelp JSON'

# Open the tar file
with tarfile.open(source_filepath, 'r') as tar:
    # List all files in the tar archive
    for member in tar.getmembers():
        # Print files names
        print(member.name) 

# Open the tar file
with tarfile.open(source_filepath, 'r') as tar:
    # Copy the Review Json file into Sepcific folder 
    tar.extract('yelp_academic_dataset_review.json', path=destination_filepath)