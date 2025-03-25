import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

p = inflect.engine()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
# convert number into words
def convert_number(text):
    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []   
    for word in temp_str:
    # if word is a digit, convert the digit
    # to numbers and append into the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
        # append the word as it is
        else:
            new_string.append(word)
    # join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str


def preprocess_text(text):
    """
    Inputs: raw text 
    Outputs: cleaned text
    lowercase, puncation removal, stopword removal, 
    number conversion, whitespace removal, stem word, 
    lemmezation(strip word to root meaning)
    """
    #lower case
    text = text.lower()
    # remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    #convert number to words/Remove words
    # text = convert_number(text) I think removing numbers is more important for this project since we cant access referenced bills
    text = re.sub(r'\d+', '', text)

    # remove whitespace
    text = " ".join(text.split())
    #stop word removal
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word not in stop_words] #text is now equal to filtered text
    #lemmazation of words
    # word_tokens = text
    lemmas = [lemmatizer.lemmatize(word=word) for word in word_tokens]
    cleaned_text = lemmas

    cleaned_text = " ".join(lemmas)
    return cleaned_text