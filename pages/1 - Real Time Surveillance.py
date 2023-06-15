import cv2
import streamlit as st
import torch
from PIL import Image
import time
import tempfile 
import pandas as pd
import numpy as np
from torchvision import models, transforms
pd.options.mode.chained_assignment = None
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet') 
from gensim.models import KeyedVectors 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import pipeline
from datetime import timedelta
import os

if 'captions' not in st.session_state: 
    st.session_state['captions'] = ""
if 'search' not in st.session_state: 
    st.session_state['search'] = 1 
if 'sussometer_threshold' not in st.session_state: 
    st.session_state['sussometer_threshold'] = 0.5 
if 'search_results' not in st.session_state: 
    st.session_state['search_results'] = [] 
# i just added search ig 

# ------------------------------------------- BACKEND ------------------------------------------
@st.cache_resource
def get_model():
    device = 0 if torch.cuda.is_available else -1
    return pipeline(model="Salesforce/blip-image-captioning-large",device=device)

def image_to_caption(_image, _model):
    return _model(_image)[0]["generated_text"]




@st.cache_data(persist=True, show_spinner=False)
def filtertext(text): 
    new_tokens = [] 
    for token in word_tokenize(text): 
        new_tokens.append(lemmatizer.lemmatize(token))
    
    #assign to globally set stopwords to a local set
    stop_words = set(stopwords.words('english')+[''])
    
    #filter the stopwords and non-alphanumeric characters from the token
    filtered_tokens = [''.join(ch.lower() for ch in token if ch in letters) for token in new_tokens if not ''.join(ch for ch in token if ch in letters).lower() in stop_words]

    return filtered_tokens 


#read in data
# print("Loading model... ") 
@st.cache_data(persist=True, show_spinner=False)
def loadValues(): 
    fl = open("sussometerData.csv", 'r') #will be formatted such that odd lines are location names and evens are tags 
    rawData = fl.readlines()
    fl.close()

    #Load data and corresponding tags 
    values = [] 
    for x in range(len(rawData)):
        #temp = rawData[x].split(',')
        #temp2 = [] 
        #for i in range(len(temp)):
        #    temp2 += filtertext(temp[i].strip()) 
        #values += temp2
        values += filtertext(rawData[x].strip()) 
    return values 

@st.cache_resource
def getVectorizer(): 
    #Load pre-trained Word2Vec model
    print("Loading vectorizer... (This is usually the longest step)") 
    vectorizer = KeyedVectors.load_word2vec_format('vectorizer.bin', binary=True)
    #vectorizer = pipeline("vectorizer", model="fse/word2vec-google-news-300")
    return vectorizer 

#print(locTags)
print("Loading complete!") 

#define function to get word similarities 
@st.cache_data(persist=True, show_spinner=False)
def word_similarities(target_word):
    distances = [] 
    for v in values:
        distances.append(vectorizer.similarity(target_word, v)) 
    #distances = vectorizer.distances(target_word, values) #ordered based on orders of vocabulary it seems
    #return (distances-np.min(distances))/(np.max(distances)-np.min(distances))
    return distances 

#function to test this 
@st.cache_data(persist=True, show_spinner=False)
def sussometer(text, threshold=st.session_state['sussometer_threshold']): #threshold is required similarity to count 
    global training
    global data
    global freqs
    t = filtertext(text)
    count = 0 
    #print(t)
    for inword in t:
        try:
            scores = word_similarities(inword)
            #print(inword)
            #print(scores) 
            c = 0 #count
            try: 
                #c *= (0.1+max(scores)) #to make sure it doesnt go to like 0
                for idx in range(len(scores)):
                    score = scores[idx]
                    if score > threshold:
                        c += 1
                        #print(score, values[idx]) 
            except:
                pass 
            count += c
        except Exception as ex:
            #word doesn't exist
            pass
    
    return count 


#-------------------------------------------- FRONTEND -----------------------------------------
keyreader = open("apikey.txt", 'r') 
openai_key = keyreader.readline().strip() 
keyreader.close

blip_model = get_model()
lemmatizer = WordNetLemmatizer()
vectorizer = getVectorizer()
#for the lists later: no. of blanks is number of topics because yes. Each topic is assigned a certain "id". 
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
values = loadValues() 


blip_model = get_model()

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

if 'placeholder' not in st.session_state:
    st.session_state["placeholder"] = st.empty()

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    # image processing
    PIL_image = Image.fromarray(frame)
    #st.image(PIL_image)
    st.session_state['captions'] = image_to_caption(PIL_image,blip_model) + "\n" + st.session_state['captions'] 
    #st.write(st.session_state['captions'])
    st.session_state["placeholder"].empty()
    st.session_state["placeholder"].text(st.session_state['captions'])

    time.sleep(1)

else:
    st.write('Stopped')
    st.session_state['captions'] = ""
    st.session_state["placeholder"].empty()