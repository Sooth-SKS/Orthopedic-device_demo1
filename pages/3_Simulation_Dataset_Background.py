# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:33:17 2022

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




st.title("Why Simulation is Needed?")
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
"""
Mobility of the fractured segments is often beneficial for the formation of a callus, but it results in substantial loading of the applied fixation device, which may cause stability, strength, or durability related issues.    
- Structural analysis is employed to assess bone and fixator deformations, stresses, and strains, which are related to the fixator durability.
- For a known fixator configuration and position relative to the bone, structural analysis of bone-fixator systems is performed using the Finite Element Method (FEM).
- Using simulation data, an optimization study can be employed to find the optimum shape and dimensions of an existing fixation device. 
"""
)
    
    
fig_col1, fig_col2, fig_col3  = st.columns([6,1,6])  

with fig_col1:
    image = Image.open('FEA model SIF-Femur asssembly.jpg')
    st.image(image, width=450,caption='Finite element (FE) model of the femur–SIF assembly')
    
    
    
with fig_col3: 
    image = Image.open('stress field_1.png')
    st.image(image, width=400,caption='Stress field of the fixator (from FEA simulation)')
    

 

with st.expander("Simulation Dataset"):
   
    df6=pd.read_csv("./DOE6.csv", 
                skiprows=4, 
                names=['Name',
                       'Bar length',
                       'Bar diameter',
                       'Bar end thickness',
                       'Radius trochanteric unit',
                       'Radius bar end',
                       'Clamp distance',
                       'Total Deformation Maximum',
                       'Equivalent Stress',
                       'P9',
                       'P10',
                       'P11',
                       'Fixator Mass'], 
                usecols=['Bar length',
                         'Bar diameter',
                         'Bar end thickness',
                         'Radius trochanteric unit',
                         'Radius bar end',
                         'Clamp distance',
                         'Total Deformation Maximum',
                         'Equivalent Stress',
                         'Fixator Mass'])
    st.markdown('The number of designpoint used for training AI model: ' + str(df6.shape[0]))
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    fig_col1, fig_col2, fig_col3  = st.columns([1,2,1]) 
    with fig_col2:
        image = Image.open('parametric model.jpg')
        st.image(image,caption='CAD model of the SIF')
    
    
     
    
    st.dataframe(df6)
    
    
