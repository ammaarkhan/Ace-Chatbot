import random 
import json 
import pickle as pkl
import numpy as np
import nltk

from tensorflow import keras
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# creating a WordNetLemmatizer() class to get the root words
lemmatizer = WordNetLemmatizer()

# loading files we made in training.py
intents = json.loads(open("intents.json").read())
words = pkl.load(open('words.pkl', 'rb'))
classes = pkl.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodelv1.h5')

def clean_up_sentences(sent):
    sentence_words = nltk.word_tokenize(sent) # returns an array of words for the sentence 
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] # returns root word for each word in array
    return sentence_words

def bagofw(sent):
    # separate words from input sentence
    sentence_words = clean_up_sentences(sent) # array of root words from input sentence
    bag = [0]*len(words) # create array of zeros same size of words array
    # if w in 'sentence_words' is in 'words' array, make bag[i] = 1 at same index as where w is in 'words' array
    for w in sentence_words: 
        for i, word in enumerate(words):
            if w == word:
                bag[i] = 1
    return np.array(bag)


def predict_class(sent):
    bow = bagofw(sent) # array of 0s and 1s, 1 representing the word is present
    res = model.predict(np.array([bow]))[0] # predict and get the result
    resss = model.predict(np.array([bow])) # ex: [[0.9838628  0.00768638 0.0084508 ]] (for input 'Hi')
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] 
    results.sort(key=lambda x: x[1], reverse=True) # sort from largest to smallest according to probability
    return_list = []
    for r in results: 
        return_list.append({'intent': classes[r[0]],
                            'probability': str(r[1])})
    return return_list
    
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Chatbot running")

while True:
    message = input("")
    if message == "end":
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
    