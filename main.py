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

# model_list = ['models/chatbotmodelv4.h5'] # 'chatbotmodelv1.h5', 'chatbotmodelv2.h5', 'chatbotmodelv3.h5'
modelName = 'models/chatbotmodelv4.h5'

# creating a WordNetLemmatizer() class to get the root words
lemmatizer = WordNetLemmatizer()

# loading files we made in training.py
courses =  json.loads(open("courses.json").read())
intents = json.loads(open("intents.json").read())
words = pkl.load(open('words.pkl', 'rb'))
classes = pkl.load(open('classes.pkl', 'rb'))
# model = load_model('chatbotmodelv1.h5')

def clean_up_sentences(sent):
    sentence_words = nltk.word_tokenize(sent.replace("'", "")) # returns an array of words for the sentence 
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words] # returns root word for each word in array
    return sentence_words

def bagofw(sent):
    testModelInput = ""
    # separate words from input sentence 
    sent = contractions.fix(sent.lower()) # fix the contractions and lower input sentence
    sentence_words = clean_up_sentences(sent) # array of root words from input sentence
    bag = [0]*len(words) # create array of zeros same size of words array
    # if w in 'sentence_words' is in 'words' array, make bag[i] = 1 at same index as where w is in 'words' array
    for inputWord in sentence_words: 
        for i, word in enumerate(words):
            if inputWord == word: # was word.lower() previously, to lower pkl words for comparison
                testModelInput = testModelInput + " " + inputWord
                bag[i] = 1
            # else:
            #     words.append(w)
    print("Model input:" + testModelInput)
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
    
def get_response(results_list, user_message):
    result = ""
    if bool(results_list) == False:
        result = random.choice(["Sorry, can't understand you. Please rephrase the question", "Please give me more info",
                "Not sure I understand, Please rephrase the question"]) 
        return result
    tag = results_list[0]['intent'] # gets the 'intent' value from the dictionary
    if (tag == 'course-major'):
        return ner_response(user_message)
    list_of_intents = intents['intents'] # retrieving json file
    for i in list_of_intents: 
        if i['tag'] == tag: # retrieving responses for that tag
            result = random.choice(i['responses'])
            break
    return result

def ner_response(user_message):
    doc = nlp2(user_message.lower())
    # print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    
    user = {}
    for ent in doc.ents:
        user.update({ent.label_: ent.text})

    print("this is the course", user['COR'])
    print("this is the major", user['MAJOR'])
    
    # direct = {"computer science": ["cosc 121", "math 100", "math 101", "cosc 211", "cosc 221", "cosc 222", "math 221", "stat 230", "cosc 320", "cosc 304", "cosc 310", "cosc 341", "cosc 499", "phil 331"], "chemistry": ["math 100", "chem 201", "chem 220", "chem 203", "chem 203", "chem 204", "chem 211", "math 200"]}
    # optional = {"computer science": [["cosc 111", "cosc 123"], {"engl 109": ["2", "engl 112", "engl 113", "engl 114", "engl 150", "engl 151", "engl 153", "engl 154","engl 155", "engl 156", "engl 203", "corh 203", "corh 205", "apsc 176", "apsc 201"]}, ["phys 111", "phys 112"]], "chemistry": [["chem 111", "chem 121"], ["chem 113", "chem 123"], ["math 101", "math 103"], {"engl 109": ["2", "engl 112", "engl 113", "engl 114", "engl 150", "engl 151", "engl 153", "engl 154","engl 155", "engl 156", "corh 203"]}, ["phys 111", "phys 112"], ["phys 121", "phys 122"]]}
    direct = courses['mandatory']
    optional = courses['optional']
    
    reply = ""
    
    if user["MAJOR"] in list(direct.keys()):
        if user['COR'] in direct[user['MAJOR']]:
            reply = "Yes " + user['COR'].upper() + " is a requirement for " + user['MAJOR'].upper() + "." 
        else:
            for type in optional[user['MAJOR']]:
                if isinstance(type, dict):
                    l=[]
                    [l.extend([k,v]) for k,v in type.items()]
                    y = [l[0]]
                    [y.extend(l[1])]
                    if user['COR'] in y:
                        reply = "Yes, take one of "+ y[0].upper() + " or " + y[1] + " of "  
                        del y[0]
                        del y[0]
                        for i in y:
                            reply += i.upper() + ", "
                else:
                    if user['COR'] in type:
                        reply = "Yes, take one of "+ type[0].upper() 
                        del type[0]
                        for i in type:
                            reply += " or " + i.upper()
    else:
        reply = "Major information not available."
            
    if reply == "":
        reply = user['COR'].upper() + " is not a requirement for " + user['MAJOR'] + " but might be used as an elective, speak with an Academic & Career Advisor for more clarity."

    return reply

print("Chatbot running\n")

print("Loading model")
model = load_model(modelName)

def update_csv():
    bar = 0
    baz = 0
    list_of_intents = intents['intents'] # retrieving json file
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Pattern", "AccIntent", "CalcIntent", "RawCalc"])
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

# for data cleaning purpose
# for i, word in enumerate(words):
#     print (str(i) + " " + word)

# to update csv results
# update_csv()

while True:
    user_message = input("\nUser: ")
    if user_message == "end":
        break
    results_list = predict_class(user_message) # [{'intent': 'greeting', 'probability': '0.9163127'}]
    bot_message = get_response(results_list, user_message) # randomly chosen response with same tag as prediction
    print("Ace: " + bot_message)