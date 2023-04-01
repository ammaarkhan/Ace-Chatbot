import streamlit as st
from streamlit_chat import message as st_message
from tensorflow import keras
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random 
import json 
import pickle as pkl
import numpy as np
import nltk
import contractions

from pathlib import Path
import spacy

from main import predict_class, get_response 

# loading files we made in training.py
intents = json.loads(open("intents.json").read())
# words = pkl.load(open('words.pkl', 'rb'))
# classes = pkl.load(open('classes.pkl', 'rb'))


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Ace Chatbot")

def generate_answer():
    user_message = st.session_state.text
    results_list = predict_class(user_message, 'models/chatbotmodelv4.h5')
    message_bot = get_response(results_list, intents)

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})

st.text_input("I'm here to help!", key="text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)  # unpacking