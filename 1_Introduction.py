# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:49:11 2022

@author: susym
"""
import streamlit as st
from PIL import Image
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

  

st.markdown("<h1 style='text-align: center; color: black'>AI based assistance for design engineers to accelerate the product development process</h1>", unsafe_allow_html=True)
st.write("Don’t waste your past design experiences. Learn from them using highly user-friendly AI assistance tool and be more creative and confident in your next design.")
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown(
"""
Properties of AI assistance tool :    
- *Say no to your local desktop. Host all your data on cloud*.  
- *Highly user-friendly to engineers. They don’t need to know coding or AI*.
- *Highly customized for the problems you solve*.
- *Low cost to initiate and to run the platform*
"""
)
 
