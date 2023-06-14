
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
import open_clip
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

# app
# title and logo
st.set_page_config(
    page_title= "App Name", 
    page_icon="",
    layout = "centered",
    initial_sidebar_state = "auto"
    )


#BACKEND STUFF ---------------------------------------------------------------------------------------------------------------------------------------------
# The following comments have bracket corresponding to the tep numbers in the workflow in the google docs 



# First, do all the preprocessing + defining functions and stuff 

# (2) define function to turn video into image frames, inputs are video (opencv format) and the time between frames (to be extracted) 

@st.cache_data(persist=True, show_spinner=False)
def getVideoFrames(vid, targetfps=1): 
    # video format is vid = cv2.VideoCapture('filename.mp4')
    success, init = vid.read()
    print(init.shape)

    fps = round(vid.get(cv2.CAP_PROP_FPS))

    imgs = [] 

    counter = fps 

    while vid.isOpened():
        success, img = vid.read()
        if not success:
            break
        if (counter >= fps): 
            imgs.append(img) 
            counter -= fps 
        counter += targetfps 
        
    return imgs

# (3) define function to do image captioning on each frame 

@st.cache_resource
def get_coca_model():
    # Create a database session object that points to the URL.
    model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model.cuda().eval()
    return model, transform

@st.cache_data
def image_to_caption(im):  # input image here
    im = im.convert("RGB")
    im = coca_transform(im).unsqueeze(0).cuda()
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = coca_model.generate(im)

    caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")[:-2]
    return caption


# (4) define function to identify whether a timestamp is suspicious (naive bayes classifier) - sussometer 

#loading stuff

#initialize lemmatizer
#print("Initializing lemmatizer and related functions... ") 
lemmatizer = WordNetLemmatizer()

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

#for the lists later: no. of blanks is number of topics because yes. Each topic is assigned a certain "id". 
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

#read in data
print("Loading model... ") 
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

@st.cache_resource
def getVectorizer(): 
    #Load pre-trained Word2Vec model
    print("Loading vectorizer... (This is usually the longest step)") 
    vectorizer = KeyedVectors.load_word2vec_format('vectorizer.bin', binary=True)
    #vectorizer = pipeline("vectorizer", model="fse/word2vec-google-news-300")
    return vectorizer 

vectorizer = getVectorizer()

# Calculate the vector representation for the keywords
print("Loading vectors... ") 
keyword_vectors = []
i = 0 
while i < len(values):
    try:
        temp = vectorizer[values[i]]
    except Exception as ex:
        name = str(ex).split("'")[1]
        values.remove(name)
        print("Vectorizer doesn't contain", name)
        continue
    keyword_vectors.append(temp) 
    i += 1 
#keyword_vectors = [vectorizer[word] for word in taglookup]


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
def sussometer(text, threshold=0.5): #threshold is required similarity to count 
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








# FRONTEND STUFF -----------------------------------------------------------------------------------------------------------------------------------------------

# GLOBAL VARIABLES
coca_model, coca_transform = get_coca_model()

#display everything 

video_type = st.sidebar.selectbox("Choose footage input mode", ["Upload footage", "Real-time footage"])

#1. Upload Video
def upload_page():

    #imports
    import streamlit as st
    import altair as alt
    import pandas as pd

    #1.A. Title
    st.write("Surveillance footage upload")
    

    #1.B. Upload Button

    #Now, the main website itself 

    # configs

    # (1) Take a video 

    st.header("App Name")
    uploaded_file = st.file_uploader("Upload your video footage here!",type=["mp4"])
    if uploaded_file is not None:
        pass
        # DO SOMETHING TO VIDEO

        # (2) turn the video into image frames - if real-time, just get frame from video. 


        # (3) do image captioning on each frame. Then, (6) generate the log 


        # (4) identify suspicious timestamps based on captions 


        # (5) generate a summary 




    #1.C. Display Summary + summary timestamp video

    #1.D. Display frames + slider




#2. Real-time Video

def realtime_page():

    #imports
    import streamlit as st
    import pandas as pd
    import altair as alt

    #2.A. Title
    st.write("Real-time surveillance footage")

    #2.B. Real-time video access feature? (bluetooth?, wifi?)

    #2.C. Suspicion Alert System

    #2.D. Display timestamps + timestamp video

    #2.E Display real time video feed



