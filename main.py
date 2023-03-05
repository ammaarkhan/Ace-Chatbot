import random 
import json 
import pickle as pkl
import numpy as np
import nltk

from tensorflow import keras
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# loading files we made
intents = json.loads(open("intents.json").read())
words = pkl.load(open('words.pkl', 'rb'))
classes = pkl.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodelv1.h5')

def clean_up_sentences(sent):
    sentence_words = nltk.word_tokenize(sent)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bagofw(sent):
    
    # separate words from input sentence
    sentence_words = clean_up_sentences(sent)
    bag = [0]*len(words)
    for w in sentence_words: 
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sent):
    bow = bagofw(sent)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
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
    