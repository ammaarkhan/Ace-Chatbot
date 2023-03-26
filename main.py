import random 
import json 
import pickle as pkl
import numpy as np
import nltk
import csv
import contractions

from tensorflow import keras
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

from pathlib import Path
import spacy

# loading ner model
output_dir=Path("NerModels")
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)

model_list = ['models/chatbotmodelv4.h5'] # 'chatbotmodelv1.h5', 'chatbotmodelv2.h5', 'chatbotmodelv3.h5'

# creating a WordNetLemmatizer() class to get the root words
lemmatizer = WordNetLemmatizer()

# loading files we made in training.py
intents = json.loads(open("intents.json").read())
words = pkl.load(open('words.pkl', 'rb'))
classes = pkl.load(open('classes.pkl', 'rb'))
# model = load_model('chatbotmodelv1.h5')

def clean_up_sentences(sent):
    sentence_words = nltk.word_tokenize(sent.replace("'", "")) # returns an array of words for the sentence 
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] # returns root word for each word in array
    return sentence_words

def bagofw(sent):
    test = ""
    # separate words from input sentence 
    sent = contractions.fix(sent.lower())
    sentence_words = clean_up_sentences(sent) # array of root words from input sentence
    bag = [0]*len(words) # create array of zeros same size of words array
    # if w in 'sentence_words' is in 'words' array, make bag[i] = 1 at same index as where w is in 'words' array
    for w in sentence_words: 
        for i, word in enumerate(words):
            if w == word.lower(): # this is to the lower the words in the pkl file
                test = test + " " + w
                bag[i] = 1
            # else:
            #     words.append(w)
    print("Model input:" + test)
    return np.array(bag)

def predict_class(sent):
    bow = bagofw(sent) # array of 0s and 1s, 1 representing the word is present
    res = model.predict(np.array([bow]))[0] # predict and get the result, of type numpy.ndarray
    # resss = model.predict(np.array([bow])) # ex: [[0.9838628 0.00768638 0.0084508 ]] (for input 'Hi')
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] # this removes the values that are below the 0.25, ex: discards [0.00768638, 0.0084508]
    results.sort(key=lambda x: x[1], reverse=True) # sort from largest to smallest according to probability
    print(results) # [[0, 0.9851791]] 
    results_list = []
    for r in results: # r is [0, 0.9851791]
        results_list.append({'intent': classes[r[0]], # classes[0]
                            'probability': str(r[1])}) # str(0.9851791)
    return results_list
    
def get_response(results_list, intents_json):
    result = ""
    if bool(results_list) == False:
        result = random.choice(["Sorry, can't understand you. Please rephrase the question", "Please give me more info",
                "Not sure I understand, Please rephrase the question"]) 
        return result
    tag = results_list[0]['intent'] # gets the 'intent' value from the dictionary
    if (tag == 'course-major'):
        return ner_response(message)
    list_of_intents = intents_json['intents'] # retrieving json file
    for i in list_of_intents: 
        if i['tag'] == tag: # retrieving responses for that tag
            result = random.choice(i['responses'])
            break
    return result

def ner_response(message):
    doc = nlp2(message)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    
    user = {}
    
    for ent in doc.ents:
        user.update({ent.label_: ent.text})
    direct = {"computer science": ["cosc 121", "cosc 240", "cosc 341"]}
    
    reply = ""
    if user['COR'] in direct[user['MAJOR']]:
        reply = "Yes " + user['COR'] + " is a requirement for " + user['MAJOR'] 
    else:
        reply = user['COR'] + " is not a requirement for " + user['MAJOR'] + " but might be used as an elective, speak with an Academic & Career Advisor for more clarity."
    
    # temp = user['COR'] + " " + user['MAJOR']
    return reply

print("Chatbot running\n")

print("Loading model")
model = load_model(model_list[0])

def update_csv():
    bar = 0
    baz = 0
    list_of_intents = intents['intents'] # retrieving json file
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pattern", "AccIntent", "CalcIntent", "RawCalc"])
        for modelName in model_list:
            writer.writerow(["", "model name:", modelName])
            for i in list_of_intents:
                for x in i['patterns']:
                    bar += 1
                    res = predict_class(x.lower()) 
                    if not res:
                        writer.writerow([x, i['tag'], "null", "no prediction"])
                        break
                    # print(x)
                    # print(i['tag'])
                    # print(res[0]['intent'])
                    # print(res)
                    writer.writerow([x, i['tag'], res[0]['intent'], res])
                    if i['tag'] == res[0]['intent']:
                        baz += 1
            writer.writerow([ "", "Accuracy", (baz/bar)*100])
            print("\nAccuracy of model: " +str(((baz/bar)*100)))
            bar = 0
            baz = 0
            print("\nResults.csv updated! \n\nWelcome to chatbot!")

# for i, word in enumerate(words):
#     print (str(i) + " " + word)

# to update csv results
# update_csv()

while True:
    message = input("\nUser: ")
    if message == "end":
        break
    results_list = predict_class(message) # [{'intent': 'greeting', 'probability': '0.9163127'}]
    final_res = get_response(results_list, intents) # randomly chosen response with same tag as prediction
    print("Ace: " + final_res)