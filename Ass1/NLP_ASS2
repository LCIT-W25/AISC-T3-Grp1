import pandas as pd
import nltk
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define slang and abbreviation dictionaries
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

# Function to handle emojis
def handle_emojis(text):
    return emoji.demojize(text)  # Converts emojis to text descriptions

# Function to handle slang words
def handle_slang(text):
    words = text.split()
    return " ".join([slang_dict.get(word, word) for word in words])

# Function to handle abbreviations
def handle_abbreviations(text):
    words = text.split()
    return " ".join([abbreviations_dict.get(word, word) for word in words])

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):  # Handle missing values
        return ""
    
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Handle emojis
    text = handle_emojis(text)
    
    # Step 3: Handle slang and abbreviations
    text = handle_slang(text)
    text = handle_abbreviations(text)
    
    # Step 4: Tokenization
    tokens = word_tokenize(text)
    
    # Step 5: Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    
    # Step 6: Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Step 7: Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

# Load dataset (update file path if needed)
df = pd.read_csv("Sentiment_Data.csv", encoding="ISO-8859-1")

# Apply text preprocessing
df['cleaned_text'] = df['Tweet'].apply(preprocess_text)

# Save cleaned dataset
df.to_csv("cleaned_sentiment_data.csv", index=False)

print("Data processing and cleaning completed successfully!")
