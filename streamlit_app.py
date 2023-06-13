
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch

# title and logo
st.title("App Name")
# st.set_page_config(page_title= “Evoke Ex-Stream App”, page_icon=”evoke_logo.png”)


st.write("Upload your video here, or take a video from your webcam!")
