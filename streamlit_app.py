
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch

# title and logo
st.title("App Name")
# st.set_page_config(page_title= “Evoke Ex-Stream App”, page_icon=”evoke_logo.png”)


# The following comments have bracket corresponding to the tep numbers in the workflow in the google docs 

# First, do all the preprocessing + defining functions and stuff 

# (2) define function to turn video into image frames, inputs are video (opencv format) and the time between frames (to be extracted) 

# (3) define function to do image captioning on each frame 



#Now, the main website itself 

# (1) Take a video 

st.write("Upload your video here, or take a video from your webcam!")







