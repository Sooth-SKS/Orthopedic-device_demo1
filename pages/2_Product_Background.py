# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:31:29 2022

@author: susym
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import collections
import warnings
import time
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")
import base64




st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(
    png_file,
    background_position="50% 2%",
    margin_top="10%",
    image_width="90%",
    image_height="",
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid="stSidebarNav"] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    margin-top: %s;
                    background-size: %s %s;
                }
            </style>
            """ % (
        binary_string,
        background_position,
        margin_top,
        image_width,
        image_height,
    )


def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )

add_logo("Soosthsayer_logo.png")




#st.markdown("<h5 style='text-align: left; color: black;'> Product background </h5>", unsafe_allow_html=True)
st.title("Selfdynamisable internal fixator (SIF)")
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
"""
SIF is used in internal fixation of long thigh bones fractures (femur fractures).   
- The SIF comprises of a bar with anti-rotating screw in the dynamic unit on one end, with two clamps with corresponding locking screws and the trochanteric unit on the opposite end with two dynamic hip screws inside. 
- Similar to other fixation devices, the SIF represents the ultimate standard in internal fixation and in the healing of fractures without mechanical failure (e.g., bending of the bar or breaking of screws).
"""
)

fig_col1, fig_col2= st.columns(2)  


with fig_col1:
    image = Image.open('real product.jpg')
    st.image(image, width=500,caption='Different components of SIF device')
    
with fig_col2:
     image = Image.open('x ray image.jpg')
     st.image(image, width=500,caption='Radiographs of a right subtrochanteric femur fracture a) before treatment b) after treatment')
