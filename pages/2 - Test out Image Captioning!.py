import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

@st.cache_resource
def get_model():
    device = 0 if torch.cuda.is_available else -1
    return pipeline(model="Salesforce/blip-image-captioning-large",device=device)

def image_to_caption(_image, _model):
    return _model(_image)[0]["generated_text"]

blip_model = get_model()

image_uploader = st.file_uploader("Upload your image caption here!", type=["jpg","png","jpeg","webp"])
if image_uploader is not None:
    image = Image.open(image_uploader)
    st.markdown("<h3> Caption: " + image_to_caption(image,blip_model) + "</h3>", unsafe_allow_html=True)
    st.image(image)