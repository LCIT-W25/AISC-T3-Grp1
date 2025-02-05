import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # Import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from PIL import Image
import numpy as np
import tensorflow as tf

class Processing:

    def __init__(self):
        pass


    def test_precessing(self, text : str):

        # Download NLTK data to the current folder
        nltk.download('stopwords', download_dir='.')
        nltk.download('punkt_tab', download_dir='.')

        text = text.lower().strip() # Convert to lowercase
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabet characters
        tokens = word_tokenize(text)  # Tokenize text into words
        stop_words = set(stopwords.words('english'))  # Set of English stopwords
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        stemmer = PorterStemmer()  # Initialize the Porter Stemmer
        tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
        return ' '.join(tokens)  # Join tokens back into a string
    
    def preprocess_image(self, image):

        img = Image.open(image)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)

        print(img_array.shape)
        print(img_array)
        # Optionally, resize the image
        return img_array / 255.0