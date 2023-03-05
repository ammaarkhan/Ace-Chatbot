import random 
import json 
import pickle as pkl
import numpy as np
import nltk
# nltk.download('punkt') # run this line first time if error shows
# nltk.download('wordnet') # run this line first time if error shows

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
        # print(documents) # for testing
        
        # adding tag to class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) # lemmatizes the word - gives root word
    for word in words if word not in ignore_letters] 
words = sorted(set(words)) # removes the duplicates

# save the words and classes lists as binary files
pkl.dump(words, open('words.pkl', 'wb'))
pkl.dump(classes, open('classes.pkl', 'wb'))


# converting words to numbers
# neural networks only work with numbers
training = []
output_empty = [0]*len(classes)

# print(documents)
# print(words)
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # print(word_patterns)
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        
    # copy of output_empty
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1 # finding index of the class to make it equal one for the training
    training.append([bag, output_row])

random.shuffle(training)
training= np.array(training)
# print(training)

# splitting data
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# model building
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compiling model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), 
                 epochs=200, batch_size=5, verbose=1)

# save the model 
model.save("chatbotmodelv1.h5", hist)

print("Training Complete!")