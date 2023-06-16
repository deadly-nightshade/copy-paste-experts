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
from datetime import timedelta
import os

# app
# title and logo
st.set_page_config(
    page_title= "Smart Surveillance", 
    page_icon="",
    layout = "wide",
    initial_sidebar_state = "auto"
    )

main_col, search_col = st.columns([3, 2], gap='medium') 

# ALL SESSION STATES
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
if 'similarity_threshold' not in st.session_state: 
    st.session_state['similarity_threshold'] = 0.6 
if 'search_results' not in st.session_state: 
    st.session_state['search_results'] = [] 
if 'video_filename' not in st.session_state:
    st.session_state['video_filename'] = ""
if 'generated_summary' not in st.session_state:
    st.session_state['generated_summary'] = ""
if 'suspicious_timestamps' not in st.session_state:
    st.session_state['suspicious_timestamps'] = ""
if 'downloaded_logs' not in st.session_state:
    st.session_state['downloaded_logs'] = ""
#os.environ['TRANSFORMERS_OFFLINE'] = 'yes'

#BACKEND STUFF ---------------------------------------------------------------------------------------------------------------------------------------------
# The following comments have bracket corresponding to the tep numbers in the workflow in the google docs 



# First, do all the preprocessing + defining functions and stuff 

# (2) define function to turn video into image frames, inputs are video (opencv format) and the time between frames (to be extracted) 

# @st.cache_data(persist=True, show_spinner=False)
def getVideoFrames(_vid, targetfps=1): 
    # video format is vid = cv2.VideoCapture('filename.mp4')
    #success, init = vid.read()

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
            c += 1 
            counter -= fps 
        counter += targetfps 
    return imgs

# (3) define function to do image captioning on each frame 

@st.cache_resource
def get_model():
    device = 0 if torch.cuda.is_available() else -1
    print(device)
    return pipeline(model="Salesforce/blip-image-captioning-large",device=device)

def image_to_caption(_image, _model):
    return _model(_image)[0]["generated_text"]

# (4) define function to identify whether a timestamp is suspicious (naive bayes classifier) - sussometer 

#loading stuff

#initialize lemmatizer

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

print("Loading complete!") 

#define function to get word similarities 
#@st.cache_data(persist=True, show_spinner=False)
def word_similarities(target_word):
    distances = [] 
    for v in loadValues():
        try: 
            #print("sussometer word similarities", target_word, v) 
            distances.append(vectorizer.similarity(target_word, v)) 
            #print(distances[-1])
        except: 
            pass 
    #distances = vectorizer.distances(target_word, values) #ordered based on orders of vocabulary it seems
    #return (distances-np.min(distances))/(np.max(distances)-np.min(distances))
    #print(distances)
    return distances 

#function to test this 
#@st.cache_data(persist=True, show_spinner=False)
def sussometer(text, threshold=st.session_state['sussometer_threshold']): #threshold is required similarity to count 
    print("Sussometer hehe")
    global training
    global data
    global freqs
    t = filtertext(text)
    print(t, threshold)
    count = 0 
    #print(t)
    for inword in t:
        try:
            scores = word_similarities(inword)
            #print(inword, scores)
            #print(inword)
            #print(scores) 
            c = 0 #count
            try: 
                #c *= (0.1+max(scores)) #to make sure it doesnt go to like 0
                for idx in range(len(scores)):
                    score = scores[idx]
                    if score > threshold:
                        print(inword, loadValues()[idx])
                        c += 1
                        #print(score, values[idx]) 
            except:
                print("Error adding score")
                pass 
            count += c
        except Exception as ex:
            #word doesn't exist
            pass
    
    print("Sus value:", count)
    return count 

# (5) define function to generate a summary of the video; summarize every sussy period, and summarize unsussy part 
# def writeSummary(captions):
    #request = ""
    # for caption in captions:

# (6) define function to tidy up the camera logs from logs
def generateLogs(logs, video_uploader, fps):
    # logs is a 2D array, each inner array is [caption, real time count, frame number]

    # 1st line - add video name, video total time, FPS used
    video_total_time = logs[-1][1]  # in seconds
    formatted_time = get_timestamp_from_seconds(video_total_time)
    
    formatted_logs = video_uploader.name + ", " + formatted_time + " , FPS used: " + str(fps) + "\n"

    #LOG FORMAT
    #TIMESTAMP
    #Video time, fps used
    #Frame 1  00:00  Caption
    
    for array in logs:
        line = "Frame " + str(array[2]) + " " + get_timestamp_from_seconds(array[1]) + "  " + array[0] + "\n"
        formatted_logs += line

    return formatted_logs

@st.cache_data(persist=True, show_spinner=False)
def get_timestamp_from_seconds(sec):
    td = timedelta(seconds=sec)
    return str(timedelta(seconds=sec))

def countTokens(text): 
    import tiktoken
    from tiktoken_ext.openai_public import ENCODING_CONSTRUCTORS 
    encFunc = ENCODING_CONSTRUCTORS['gpt-4.5-turbo-16k']
    encDict = encFunc() 
    enc = tiktoken.Encoding(encDict['name'],
         pat_str         = encDict['pat_str'        ],
         mergeable_ranks = encDict['mergeable_ranks'],
         special_tokens  = encDict['special_tokens' ])
    return len(enc.encode(text))

def genSummary(captions, suslist):
    print("Loading summary... ")
    import openai
    keyreader = open("apikey.txt", 'r') 
    openai.api_key = keyreader.readline().strip() 
    keyreader.close()

    messages = [{"role": "system", "content": "You are generating a brief summary of a video given a list of captions. In your reply, only state the summary and nothing else, revising it with each new prompt if required, but make sure to keep the content from previous replies, and summarize it briefly, removing unnecessary detail"}]
    request = "Create a brief summary in chronological order of this list of captions:\n"
    c = len(request) 
    incr = 0 
    i = 0 
    res = "" 
    print(captions)
    while(i < len(captions)):
        while (i < len(captions)):
            incr = len(captions[i] + "\n") 
            if (c+incr) > 2000: 
                break 
            request += captions[i] + '\n' 
            c += incr
            i+=1
            print(i)
        print("Submitted:", c)
			
        # post the request 
        messages.append({"role":"user", "content":request})
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)
        res = chat.choices[0].message.content
        messages.append({"role":"assistant", "content":res})
            
        request = "Continuing the list of captions:\n"
        c = len(request) 
        general_summary = res 
	
    print("General summary:", general_summary)
    
    if len(suslist) == 0: 
        print("No sus list :(")
        return (general_summary, "") 

    # now, you want it to focus on the suspicious ones 
    request = "Focus on the following frames in which suspicious events may have occurred. Group frames close to each other as the same activity;\nFrames " # global variable sus_frames = [] 
    for i in suslist: 
        request += str(i) + ', ' 
    messages.append({"role":"user", "content": request})
    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=messages)
    sus_summary = chat.choices[0].message.content
	
    print("Sus summary:", sus_summary)

    return (general_summary, sus_summary) 
	
def similar_meaning(query, sentence, threshold=0.6): 
    for word in sentence:
        if (vectorizer.similarity(query, word) >= threshold): 
            return True 
    return False 

# FRONTEND STUFF -----------------------------------------------------------------------------------------------------------------------------------------------

# GLOBAL VARIABLES
blip_model = get_model()
lemmatizer = WordNetLemmatizer()
vectorizer = getVectorizer()
#for the lists later: no. of blanks is number of topics because yes. Each topic is assigned a certain "id". 
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
values = loadValues() 

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

    uploaded_file = st.file_uploader("Upload your video footage here!",type=["mp4"],accept_multiple_files=False, key="uploaded_file")
    if uploaded_file is not None:
        #pass
        # DO SOMETHING TO VIDEO

        if uploaded_file.name == st.session_state["video_filename"]:
            # Generate download button
            # text_file_name = st.session_state['uploaded_file'].name + ".txt"

            text_file_name = st.session_state['video_filename'] + ".txt"
            with open(text_file_name, 'w') as text_file:
                text_file.write(st.session_state['downloaded_logs'])

            st.download_button( 
                label="Download the generated logs file as a txt",
                data = open(text_file_name, 'r'),
                file_name=text_file_name,
                mime='text/txt',
             )            



            # displayed Summary
            st.header("Summary")
            st.write(st.session_state['generated_summary'][0])
            st.write("Note the following frame(s) in which suspicious activities have occured:\n")
            st.write(st.session_state['generated_summary'][1])
            st.header("Suspicious occurences timestamps")
            for i in st.session_state['suspicious_timestamps']:
                st.text(st.session_state['logs'][i-1])
                #show video feed that starts 5 seconds b4 timestamp, and show brief summary of captions within the timeframe of plus-minus 10 seconds from timestamp

            playVideoPage()
        else:

        # (2) turn the video into image frames - if real-time, just get frame from video. 

        #targetfps = 1 

            st.session_state["video_filename"] = uploaded_file.name

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

            generateDownloadButton()

            displaySummary()
            playVideoPage() 

def susList(): 
    filtered = [] 
    numbers = [] 
    print("Generating sus list...")
    print(st.session_state['logs'])
    for f in st.session_state['logs']: 
        print(f)
        if (sussometer(f[0], 0.5) > 0): 
            filtered.append(f) 
            numbers.append(f[2])
            print("Appended")
    #st.text('\n'.join([i for i in st.session_state['logs'] if i[0].contains(st.session_state['search'])]))
    return numbers

def generateDownloadButton():

    # (4.1) Generate Logs
    logs = generateLogs(st.session_state['logs'], st.session_state['uploaded_file'], st.session_state['targetfps'])
    st.text(logs)
    st.session_state['downloaded_logs'] = logs

    print(st.session_state['logs'])

    text_file_name = st.session_state['uploaded_file'].name + ".txt"


    with open(text_file_name, 'w') as text_file:
        text_file.write(logs)

    st.download_button( 
        label="Download the generated logs file as a txt",
        data = open(text_file_name, 'r'),
        file_name=text_file_name,
        mime='text/txt',
    )
            
    text_file.close()

def displaySummary():
    #1.C. Display Summary + summary timestamp video
    print("111111111111111111111111111111111111111")
    tempSummTimestamps = susList() 
    print("222222222222222222222222222222222222222")
    tempSumm = genSummary([i[0] for i in st.session_state['logs']], tempSummTimestamps) #this should be a string
    st.session_state['generated_summary'] = tempSumm
    st.session_state['suspicious_timestamps'] = tempSummTimestamps

    #tempSummTimestamps = st.session_state['search_results'] #this should be an array
    st.header("Summary")
    st.write(tempSumm[0])
    st.write("Note the following frame(s) in which suspicious activities have occured:\n")
    st.write(tempSumm[1])
    st.header("Suspicious occurences timestamps")
    for i in tempSummTimestamps:
        st.text(st.session_state['logs'][i-1])
        #show video feed that starts 5 seconds b4 timestamp, and show brief summary of captions within the timeframe of plus-minus 10 seconds from timestamp

def playVideoPage(): 
    
    #1.D. Display frames + slider
    if 'videoplayer' not in st.session_state: 
        st.session_state['videoplayer'] = st.empty() 
    print("trying my best to RUN THIS REAL I MADE THE SLIDERRRR")
    updateVideo() 
    st.session_state['current_video_time'] = round(st.slider("Video time: ", 0.0, len(st.session_state['captions']) / st.session_state['targetfps'], 1/st.session_state['targetfps']) / st.session_state['targetfps'])


def load_searchbar(): 
    print("Loading search bar")
    # search thing 
    st.session_state['search'] = st.text_input("Search timetamp by keywords", value="")

    #sussometer slider 
    st.session_state['sussometer_threshold'] = st.slider("Sussometer threshold (suggest >= 0.5)", 0.0, 1.0, 0.5)
    
    #similarity threshold
    st.session_state['similarity_threshold'] = st.slider("Similarity threshold (1 is exact match)", 0.0, 1.0, 0.6)

    if 'search_res_display' not in st.session_state: 
        st.session_state['search_res_display'] = st.empty() 
    
    #do filter thingy 
    updateSearch() 
    

    
def updateVideo(): 
    st.session_state['videoplayer'].empty() 
    with st.session_state['videoplayer']: 
        img  = st.session_state['img_caption_frames'][st.session_state['current_video_time']]
        print(img)
        img = Image.fromarray(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB))
        st.image(img) 

def word_sentence_similarities(target_word, sentence, threshold):
    for word in sentence.split():
        print("word sentence similarities:", target_word, word)
        try: 
            if (vectorizer.similarity(target_word, word) >= threshold): 
                print("Word sentence similarities match:", target_word, word)
                return True  
        except: 
            print("word sentence similarities error:", target_word, word)
    #distances = vectorizer.distances(target_word, values) #ordered based on orders of vocabulary it seems
    #return (distances-np.min(distances))/(np.max(distances)-np.min(distances))
    return False 

def updateSearch(): 
    st.session_state['search_res_display'].empty() 
    filtered = [] 
    numbers = [] 
    for f in st.session_state['logs']: 
        matched = False 
        for word in st.session_state['search'].split(): 
            if word_sentence_similarities(word, f[0], st.session_state['similarity_threshold']): 
                matched = True 
                break 
        if matched and (sussometer(f[0], st.session_state['sussometer_threshold']) > 0): 
            filtered.append(f) 
            numbers.append(f[2])
    #st.text('\n'.join([i for i in st.session_state['logs'] if i[0].contains(st.session_state['search'])]))
    st.session_state['search_results'] = numbers 
    #res = "" 
    logs = []
    if st.session_state["uploaded_file"] is not None:
        logs = generateLogs(st.session_state['logs'], st.session_state["uploaded_file"], st.session_state["targetfps"]).split('\n') 
    
    with st.session_state['search_res_display']: 
        res = "<p>"
        for i in numbers: 
            res += (logs[i+1]) + '<br>' 
        res += "</p>" 
        st.markdown(res, unsafe_allow_html=True)


with main_col: 
    upload_page()
with search_col: 
    load_searchbar() 

