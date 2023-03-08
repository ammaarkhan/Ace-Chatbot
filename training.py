import random 
import json 
import pickle as pkl
import numpy as np
import nltk
# nltk.download('punkt') # run this line first time if error shows
# nltk.download('wordnet') # run this line first time if error shows

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Bidirectional
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
        word_list = nltk.word_tokenize(pattern) # returns an array of words for each entry in the pattern, one by one for every pattern in patterns ex: ['hello']
        # print(word_list) # for testing
        words.extend(word_list) # adding the word_list output for each entry  of root words to words array
        # print("this is the words: {}".format(words)) # for testing, ex: ['hello', 'hi', 'hey', 'what', "'s", 'up', '?'] (after few iterations of loop)
        
        # linking patterns with respective tags and adding to documents list as a tuple
        documents.append(((word_list), intent['tag']))
        # print(documents) # for testing ex: [(['hello'], 'greeting')]\n[(['hello'], 'greeting'), (['hi'], 'greeting')]
        
        # adding tag to class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(classes) # for testing, ex: ['greeting', 'name', 'age']
words = [lemmatizer.lemmatize(word) # lemmatizes the word - gives root word
    for word in words if word not in ignore_letters] 
words = sorted(set(words)) # removes the duplicates
# print(words) # for test, ex: ["'s", 'age', 'are', 'call'.......]

# save the words and classes lists as binary files
pkl.dump(words, open('words.pkl', 'wb'))
pkl.dump(classes, open('classes.pkl', 'wb'))


# converting words to numbers
# neural networks only work with numbers
training = []
output_empty = [0]*len(classes) # [0, 0, 0]

for doc in documents: 
    # print(doc) # ex: (['hello'], 'greeting')
    bag = []
    word_patterns = doc[0] # ex: ['hello']
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # print(word_patterns) # ^^ does nothing for the current intents we have
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
        # print(bag)
        
    # copy of output_empty
    tag_row = list(output_empty)
    tag_row[classes.index(doc[1])] = 1 # finding index of the class to make it equal one for the training
    training.append([bag, tag_row]) # adding array of 0s, 1s - 1 reps word,  array of 0s, 1s - 1 reps the tag
    # print(training) # example: [[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0]]]

random.shuffle(training)
training= np.array(training)
# print(training)

# splitting data into x and y values: arrays of words and arrays of tags
train_x = list(training[:, 0]) # ex: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(train_x)
train_y = list(training[:, 1]) # ex: [1, 0, 0]
# print(train_y)

# model building
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compiling model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), 
                 epochs=60, batch_size=5, verbose=1)

# save the model 
model.save("chatbotmodelv2.h5", hist)

print(model.summary())

print("Training Complete!")