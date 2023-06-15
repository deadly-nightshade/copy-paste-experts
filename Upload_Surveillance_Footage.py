import tempfile 
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch
import cv2
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
openai_key = "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"

# app
# title and logo
st.set_page_config(
    page_title= "Smart Surveillance", 
    page_icon="",
    layout = "centered",
    initial_sidebar_state = "auto"
    )

# ALL SESSION STATES
if 'videoplayer' not in st.session_state: 
    st.session_state['videoplayer'] = st.empty() 
if 'current_video_time' not in st.session_state: 
    st.session_state['current_video_time'] = 0 
if 'img_caption_frames' not in st.session_state: 
    st.session_state['img_caption_frames'] = [] 
if 'captions' not in st.session_state: 
    st.session_state['captions'] = [] 
if 'logs' not in st.session_state: 
    st.session_state['logs'] = [] 
if 'targetfps' not in st.session_state: 
    st.session_state['targetfps'] = 1 
if 'search' not in st.session_state: 
    st.session_state['search'] = 1 
if 'sussometer_threshold' not in st.session_state: 
    st.session_state['sussometer_threshold'] = 0.5 


#BACKEND STUFF ---------------------------------------------------------------------------------------------------------------------------------------------
# The following comments have bracket corresponding to the tep numbers in the workflow in the google docs 



# First, do all the preprocessing + defining functions and stuff 

# (2) define function to turn video into image frames, inputs are video (opencv format) and the time between frames (to be extracted) 

# @st.cache_data(persist=True, show_spinner=False)
def getVideoFrames(_vid, targetfps=1): 
    # video format is vid = cv2.VideoCapture('filename.mp4')
    #success, init = vid.read()
    #print(init.shape)

    fps = round(_vid.get(cv2.CAP_PROP_FPS))

    imgs = [] 

    counter = fps 
    c = 0 
    while _vid.isOpened():
        success, img = _vid.read()
        if not success:
            break
        if (counter >= fps): 
            imgs.append([img, c]) 
            counter -= fps 
        counter += targetfps 
    return imgs

# (3) define function to do image captioning on each frame 

@st.cache_resource
def get_model():
    device = 0 if torch.cuda.is_available else -1
    return pipeline(model="Salesforce/blip-image-captioning-large",device=device)

def image_to_caption(_image, _model):
    return _model(_image)[0]["generated_text"]

# (4) define function to identify whether a timestamp is suspicious (naive bayes classifier) - sussometer 

#loading stuff

#initialize lemmatizer
#print("Initializing lemmatizer and related functions... ") 

#define filter text function using lemmatizer 
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

# (5) define function to generate a summary of the video; summarize every sussy period, and summarize unsussy part 


# def writeSummary(captions):

#     request = ""
#     for caption in captions:
        



# FRONTEND STUFF -----------------------------------------------------------------------------------------------------------------------------------------------

# GLOBAL VARIABLES
blip_model = get_model()
lemmatizer = WordNetLemmatizer()
vectorizer = getVectorizer()
#for the lists later: no. of blanks is number of topics because yes. Each topic is assigned a certain "id". 
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
values = loadValues() 

#display everything 

video_type = st.sidebar.selectbox("Choose footage input mode", ["Upload footage", "Real-time footage"])


#1. Upload Video

def upload_page():

    #imports
    import streamlit as st
    import altair as alt
    import pandas as pd

    #1.A. Title
    st.title("Surveillance footage upload")
    

    #1.B. Upload Button

    #Now, the main website itself 

    # configs

    # (1) Take a video 

    st.header("App Name")
    uploaded_file = st.file_uploader("Upload your video footage here!",type=["mp4"],accept_multiple_files=False)
    if uploaded_file is not None:
        #pass
        # DO SOMETHING TO VIDEO

        # (2) turn the video into image frames - if real-time, just get frame from video. 

        #targetfps = 1 

        temp = tempfile.NamedTemporaryFile(delete=False) 
        temp.write(uploaded_file.read()) 

        vid = cv2.VideoCapture(temp.name)
        st.session_state['img_caption_frames'] = getVideoFrames(vid, st.session_state['targetfps'])

        

        #vid = cv2.VideoCapture(temp.name) 
        #success, frame = vid.read() 
        
        st.session_state['current_video_time'] = 0 

        # (3) do image captioning on each frame. Then, (6) generate the log 

        c = 0
        progress_bar = st.progress(0, text="Loading frames from video. Please wait...")
        for caption_frame in st.session_state['img_caption_frames']: 
            # TODO show progress bar!

            PIL_image = Image.fromarray(caption_frame[0])
            caption = image_to_caption(PIL_image, blip_model) 
            print(caption)
            st.session_state['captions'].append(caption) 
            st.session_state['logs'].append([caption, c / st.session_state['targetfps'], c]) #caption, real time, frame number 
            c += 1 
            progress_bar.progress(c/len(st.session_state['img_caption_frames']), text="Loading frames from video. Please wait...")

        progress_bar.empty()
        # (4) identify suspicious timestamps based on captions 

        st.write(st.session_state['logs'])

        # (5) generate a summary


        playVideoPage() 


def playVideoPage(): 

    #1.C. Display Summary + summary timestamp video

    tempSumm = Summary #this should be a string
    tempSummTimestamps = SummaryTimestamps #this should be an array
    st.header("Summary")
    st.write(tempSumm)
    st.header("Suspicious occurences timestamps")
    for i in range(len(tempSummTimestamps)):
        st.write(tempSummTimestamps[i])
        #show video feed that starts 5 seconds b4 timestamp, and show brief summary of captions within the timeframe of plus-minus 10 seconds from timestamp


    #1.D. Display frames + slider

    st.session_state['current_video_time'] = round(st.slider("Video time: ", 0.0, len(st.session_state['captions']) / st.session_state['targetfps'], 1/st.session_state['targetfps']) / st.session_state['targetfps'])
    updateVideo() 

    # search thing 
    st.session_state['search'] = st.text_input("Search timetamp by keywords", value="")
    updateSearch() 

    #sussometer slides 
    st.session_state['sussometer_threshold'] = st.slider("Sussometer threshold (suggest >= 0.5)", 0, 1, 0.05)

    #do filter thingy 

    
def updateVideo(): 
    st.session_state['videoplayer'].image(st.session_state['img_caption_frames'][st.session_state['current_video_time']]) 

def updateSearch(): 
    st.text('\n'.join([i for i in st.session_state['logs'] if i[0].contains(st.session_state['search'])]))

#2. Real-time Video

def realtime_page():

    #imports
    import streamlit as st
    import pandas as pd
    import altair as alt

    #2.A. Title
    st.title("Real-time surveillance footage")

    #2.B. Real-time video access feature? (bluetooth?, wifi?)
    if st.button("Connect real-time video feed"):
        st.write("")
        #st.write is just placeholder.
        #Here we add the connection method and stuff

    #2.C. Suspicion Alert System
    if sussometrics: #sussometrics is a bool that is true when something sus occurring
        tempSumm = RTsummary #string of summary of what happened using captions at those few frames
        st.warning(str("Something sussy!\n"+tempSumm))

    #2.D. Display timestamps + timestamp video
    RTtempSumm = RTSummary #this should be a string
    RTtempSummTimestamps = RTSummaryTimestamps #this should be an array
    st.header("Summary")
    st.write(RTtempSumm)
    st.header("Suspicious occurences timestamps")
    for i in range(len(RTtempSummTimestamps)):
        st.write(RTtempSummTimestamps[i])
        #show video feed that starts 5 seconds b4 timestamp, and show brief summary of captions within the timeframe of plus-minus 10 seconds from timestamp


    #2.E Display real time video feed
    #yes.




if video_type=="Upload footage":
    upload_page()
else:
    realtime_page()


