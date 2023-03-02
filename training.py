import random 
import json 
import pickle as pkl
import numpy as np
import nltk
# nltk.download('punkt')

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

# creating a WordNetLemmatizer() class to get the root words 
lemmatizer = WordNetLemmatizer()

# read the json intents file
intents = json.loads(open("intents.json").read())

# create empty lists to store data
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ",", "."]
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # separating words from patterns
        word_list = nltk.word_tokenize(pattern) # returns an array of root words for each entry in the pattern, one by one
        # print(word_list) # for testing
        words.extend(word_list) # adding the word_list output for each entry  of root words to words array
        # print("this is the words: {}".format(words)) # for testing
        
        # linking patterns with respective tags and adding to documents list as a tuple
        documents.append(((word_list), intent['tag']))
        # print(type(documents[0])) # for testing
        # adding tag to class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])