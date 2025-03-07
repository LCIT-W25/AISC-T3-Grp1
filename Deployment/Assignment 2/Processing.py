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
import emoji

class Processing:

    def __init__(self):
        pass # will create the class instance

    # Slang and abbreviation dictionaries
    slang_dict = {
            "brb": "be right back",
            "lol": "laugh out loud",
            "idk": "I don't know",
            "smh": "shaking my head",
            "pls": "please",
            "btw": "by the way",
            "u": "you",
            "r": "are"
        }

    abbreviations_dict = {
            "u": "you",
            "r": "are",
            "pls": "please",
            "btw": "by the way"
        }


    def test_precessing(self, text : str):

        # Download NLTK data to the current folder
        nltk.download('stopwords', download_dir='.')
        nltk.download('punkt_tab', download_dir='.')

        # Step 1: Handle emojis
        text = self.handle_emojis(str(text))
        
        # Step 2: Handle slang and abbreviations
        text = self.handle_slang(text)
        text = self.handle_abbreviations(str(text))
        
        # Step 3: Tokenize the text and convert it to lowercase
        tokens = word_tokenize(str(text).lower())  # Convert to lowercase and tokenize
        
        # Step 4: Get the set of stopwords
        stop_words = set(stopwords.words('english'))
        
        # Step 5: Initialize the stemmer
        stemmer = PorterStemmer()
        
        # Step 6: Remove stopwords, non-alphabetic and non-numeric tokens, and apply stemming
        return [stemmer.stem(word) for word in tokens if word.isalpha() or word.isdigit() and word not in stop_words]
    

    # Function to handle emojis
    def handle_emojis(self, text):
        return emoji.demojize(text)  # Converts emojis to text descriptions like ":smiling_face_with_sunglasses:"

    # Function to handle slang words
    def handle_slang(self, text):
        words = text.split()
        return " ".join([self.slang_dict.get(word, word) for word in words])

    # Function to handle abbreviations
    def handle_abbreviations(self, text):
        words = text.split()
        return " ".join([self.abbreviations_dict.get(word, word) for word in words])

    
    def preprocess_image(self, image):

        img = Image.open(image)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)

        print(img_array.shape)
        print(img_array)
        # Optionally, resize the image
        return img_array / 255.0

    def get_tweet_vector(self, words, model, vector_size=200):
        vec = np.zeros(vector_size)
        count = 0
        for word in words:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        if count > 0:
            vec /= count  # Average word vectors
        return vec

    
        