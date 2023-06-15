import cv2
import streamlit as st
import torch
from PIL import Image
from transformers import pipeline
import time

if 'captions' not in st.session_state: 
    st.session_state['captions'] = ""

# ------------------------------------------- BACKEND ------------------------------------------
@st.cache_resource
def get_model():
    device = 0 if torch.cuda.is_available else -1
    return pipeline(model="Salesforce/blip-image-captioning-large",device=device)

def image_to_caption(_image, _model):
    return _model(_image)[0]["generated_text"]


#-------------------------------------------- FRONTEND -----------------------------------------
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