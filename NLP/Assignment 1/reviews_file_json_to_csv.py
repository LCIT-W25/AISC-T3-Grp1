import pandas as pd

# Set the display option to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Specify the path to the JSON file
json_file_path =  r'C:\Users\kiran\Downloads\Yelp JSON\yelp_academic_dataset_review.json'

# Read the JSON file in chunks
chunksize = 100000

# No.of chunks with each of chunk size 
chunk_count = 1

# Split Json file with each chunk contains chunksize rows (since file format is "one JSON-object per-line" mentioned in Yelp Doc)
json_chunks = pd.read_json(json_file_path, lines=True, chunksize=chunksize)

# Created Veriable of type dataframe
df = pd.DataFrame()

# Process each chunk
for chunk in json_chunks:

    # Concatenate vertically (stack rows)
    df = pd.concat([df, chunk], axis=0)

    # Reset the index to have a continuous index after concatenation
    df.reset_index(drop= True, inplace=True)

    if(chunk_count == 1) :
        break
    else : 
        chunk_count = chunk_count - 1


print(df.shape)
print(df.head())
# Save to an Excel file
df.to_csv(r'C:\Users\kiran\Downloads\Yelp JSON\reviews.csv', index=False)
print("=========== File Created =============")