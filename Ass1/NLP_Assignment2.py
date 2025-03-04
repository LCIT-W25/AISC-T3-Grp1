import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import emoji

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

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

# Function to handle emojis
def handle_emojis(text):
return emoji.demojize(text)

# Function to handle slang words
def handle_slang(text):
words = text.split()
return " ".join([slang_dict.get(word, word) for word in words])

Function to handle abbreviations
def handle_abbreviations(text):
words = text.split()
return " ".join([abbreviations_dict.get(word, word) for word in words])

# Function to preprocess the text
def preprocess_stopwords_stemming(text):
  
text = handle_emojis(str(text))
text = handle_slang(text)
text = handle_abbreviations(str(text))
tokens = word_tokenize(str(text).lower()) # Convert to lowercase and␣tokenize
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
return [stemmer.stem(word) for word in tokens if word.isalpha() or word.isdigit() and word not in stop_words]





















