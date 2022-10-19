# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:25:17 2022

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
#df6.head()






X=df6.values[:,:6]
y=df6.values[:,6:]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .2)



model1 =RandomForestRegressor()
model2 = RandomForestRegressor()
model3 = RandomForestRegressor()



model1.fit(X_train, y_train[:,0])
y_pred_test_1 = model1.predict(X_test)


model2.fit(X_train, y_train[:,1])
y_pred_test_2 = model2.predict(X_test)


model3.fit(X_train, y_train[:,2])
y_pred_test_3 = model3.predict(X_test)



#with st.expander("Design performance prediction"):
#st.markdown("<h6 style='text-align: left; color: black;'> Design performance prediction </h6>", unsafe_allow_html=True)
#st.write("Based on the designer inputs, the design performance can be predicted. Here, the design performance parameters are total maximum deformation, equivalent stress, and fixator mass")
# create columns for the chars
#st.markdown("<hr/>", unsafe_allow_html=True)

#valid = st.sidebar.radio(label = "", options = ['Performance prediction', 'Parameteres sensitivity'])
    

#if valid == 'Performance prediction':
st.title("Design Performance Predictions")   
st.write("The design performances are: 1) Total Deformation Maximum 2) Equivalent Stress 3) Fixator Mass")
    #st.markdown("Input Parameters")
st.markdown("<hr/>", unsafe_allow_html=True) 
    
fig_col0, fig_col1, fig_col2, fig_col3, fig_col4, fig_col5 = st.columns(6)
with fig_col0:
    a = st.slider('Bar length', min_value=100, max_value=250, step=10)
with fig_col1:
    b = st.slider('Bar diameter', min_value=8.0, max_value=10.0, step=0.1)
with fig_col2:
    c = st.slider('Bar end thickness', min_value=4.0, max_value=6.5, step=0.1)
with fig_col3:
    d  = st.slider('Radius trochanteric unit', min_value=3.0, max_value=10.0, step=0.1)
with fig_col4:
    e  = st.slider('Radius bar end', min_value=6.0, max_value=10.0, step=0.1)
with fig_col5:
    f  = st.slider('Clamp distance', min_value=1.0, max_value=28.0, step=0.5)
    
st.markdown("<hr/>", unsafe_allow_html=True) 
    #st.markdown("Output performances")
input_data = np.array([a,b,c,d,e,f]).reshape(1,-1)
fig_col1, fig_col2, fig_col3 = st.columns(3)
    
with fig_col1:
    y_pred_1 = model1.predict(input_data)
    fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = round(y_pred_1[0],2),
            mode = "gauge+number+delta",
            title = {'text': "Total Deformation Maximum"},
            delta = {'reference': 18, 'increasing': {'color': "red"},'decreasing': {'color': "green"}},
            gauge = {'axis': {'range': [0, 20]},
                                 'bar': {'color': "black"},
                                 'steps' : [
                                         {'range': [0, 10], 'color': "lightgreen "},
                                         {'range': [10, 15], 'color': "yellow"},
                                         {'range': [15, 20], 'color': "red"}],
                                          'threshold' : {'line': {'color': "darkblue", 'width': 6}, 'thickness': 0.75, 'value': 18}}))
    fig.update_layout(autosize=False, width=350,height=400)
    st.write(fig)
    
with fig_col2:
    y_pred_2 = model2.predict(input_data)
    fig = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = round(y_pred_2[0],2),
                mode = "gauge+number+delta",
                title = {'text': "Equivalent Stress"},
                delta = {'reference': 600, 'increasing': {'color': "red"},'decreasing': {'color': "green"}},
                gauge = {'axis': {'range': [0, 700]}, 'bar': {'color': "black"},
                                  'steps' : [
                                          {'range': [0, 200], 'color': "lightgreen "},
                                          {'range': [200, 500], 'color': "yellow"},
                                          {'range': [500, 800], 'color': "red"}],
                                          'threshold' : {'line': {'color': "darkblue", 'width': 4}, 'thickness': 0.75, 'value': 600}}))
    fig.update_layout(autosize=False, width=350, height=400)
    st.write(fig)
    
    
with fig_col3:
    y_pred_3 = model3.predict(input_data)
    fig = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = round(y_pred_3[0],2),
                mode = "gauge+number+delta",
                title = {'text': "Fixator Mass"},
                delta = {'reference': 0.25, 'increasing': {'color': "red"},'decreasing': {'color': "green"}},
                gauge = {'axis': {'range': [0, 0.5]},
                                  'bar': {'color': "black"},
                                  'steps' : [
                                          {'range': [0, 0.1], 'color': "lightgreen "},
                                          {'range': [0.1, 0.3], 'color': "yellow"},
                                          {'range': [0.3, 0.6], 'color': "red"}],
                                          'threshold' : {'line': {'color': "darkblue", 'width': 4}, 'thickness': 0.75, 'value': 0.25}}))
    fig.update_layout(autosize=False,width=350, height=400)
    st.write(fig)
        
    

