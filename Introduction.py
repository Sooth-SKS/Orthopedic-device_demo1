# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:49:11 2022

@author: susym
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time
from PIL import Image
import plotly.graph_objects as go
import matplotlib.pylab as plt
from plotly.subplots import make_subplots



st.set_page_config(layout="wide")

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

X=df6.values[:,:6]
y=df6.values[:,6:]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .2,random_state = 4)

model1 =RandomForestRegressor(random_state = 4)
model2 = RandomForestRegressor(random_state = 4)
model3 = RandomForestRegressor(random_state = 4)

model1.fit(X_train, y_train[:,0])
model2.fit(X_train, y_train[:,1])
model3.fit(X_train, y_train[:,2])

y_pred_test_1 = model1.predict(X_test)
y_pred_test_2 = model2.predict(X_test)
y_pred_test_3 = model3.predict(X_test)

with st.sidebar.container():
    image = Image.open('Soosthsayer_logo.png')
    st.image(image, use_column_width=True)

option = st.sidebar.selectbox('',('Introduction', 'Product Background', 'Simulation Dataset Background', 'Training and Validation','Performance Prediction'))
#st.sidebar.write('You selected:', option)   

if option=="Introduction":

    st.markdown("<h1 style='text-align: center; color: black'>AI based assistance for design engineers to accelerate the product development process</h1>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.write("Don’t waste your past design experiences. Learn from them using highly user-friendly AI assistance tool and be more creative and confident in your next design.")
    st.markdown("<hr/>", unsafe_allow_html=True)

 
elif option=="Product Background":
    
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

elif option=="Simulation Dataset Background":
    
    st.title("Why Simulation is Needed?")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
           """
           Mobility of the fractured segments is often beneficial, but it results in substantial loading of the applied fixation device, which may cause stability, strength, or durability related issues.    
           - To assess bone and fixator deformations, stresses, and strains, which are related to the fixator durability, structural analysis is performed.
           - The Structural analysis of bone-fixator systems is performed using the Finite Element Analysis (FEA).
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
        st.markdown('The number of designpoint used for training AI model: ' + str(df6.shape[0]))
        st.markdown("<hr/>", unsafe_allow_html=True)
    
        fig_col1, fig_col2, fig_col3  = st.columns([1,2,1]) 
        with fig_col2:
                image = Image.open('parametric model.jpg')
        st.image(image,caption='CAD model of the SIF')
        st.dataframe(df6)
        
        
elif option=="Training and Validation":        
        
 
    def val_error():
        st.title("Validation against the test simulation data") 
        st.markdown('The number of designpoint used for validating the AI model: ' + str(X_test.shape[0]))
        st.markdown("<hr/>", unsafe_allow_html=True)    
        

        con_1 = np.concatenate((y_test[:,0].reshape(-1,1), y_pred_test_1.reshape(-1,1)), axis=1)
        con_2 = np.concatenate((y_test[:,1].reshape(-1,1), y_pred_test_2.reshape(-1,1)), axis=1)
        con_3 = np.concatenate((y_test[:,2].reshape(-1,1), y_pred_test_3.reshape(-1,1)), axis=1)

        df_val_1 = pd.DataFrame(con_1, columns=['test', 'pred'])
        df_val_2 = pd.DataFrame(con_2, columns=['test', 'pred'])
        df_val_3 = pd.DataFrame(con_3, columns=['test', 'pred'])

        df_val_1['% Error(Total Deformation Maximum)'] = abs(df_val_1['pred']-df_val_1['test'])*100/df_val_1['test'] 
        df_val_2['% Error(Equivalent Stress)'] = abs(df_val_2['pred']-df_val_2['test'])*100/df_val_2['test'] 
        df_val_3['% Error(Fixator Mass)'] = abs(df_val_3['pred']-df_val_3['test'])*100/df_val_3['test'] 

        df_val_all = pd.concat([df_val_1,df_val_2,df_val_3],axis = 1).drop(['test', 'pred'],axis=1)
        df_X_test = pd.DataFrame(X_test[0:6], columns=['Bar length','Bar diameter','Bar end thickness','Radius trochanteric unit','Radius bar end','Clamp distance'])

        df_val_error = pd.merge(df_X_test,df_val_all,left_index=True, right_index=True)

        a = df_val_error['% Error(Total Deformation Maximum)'].mean()
        b = df_val_error['% Error(Equivalent Stress)'].mean()
        c = df_val_error['% Error(Fixator Mass)'].mean()
        
        a_max = df_val_error['% Error(Total Deformation Maximum)'].max()
        b_max = df_val_error['% Error(Equivalent Stress)'].max()
        c_max = df_val_error['% Error(Fixator Mass)'].max()
        
        
         
        fig01, kpi1,  kpi2,  kpi3 = st.columns([0.2,1,1,1])
        
        kpi1.metric("Error in Total Deformation Maximum(avg/max)",str(round(a,2))+'/' + str(round(a_max,2)))
         
        kpi2.metric("Error in Equivalent Stress(avg/max)",str(round(b,2))+'/'+str(round(b_max,2)))
          
        kpi3.metric("Error in Fixator Mass(avg/max)", str(round(c,2))+'/'+str(round(c_max,2)))
        
        #st.markdown("<hr/>", unsafe_allow_html=True)
        
        fig = make_subplots(rows=1, cols=3,subplot_titles=("Total Deformation Maximum", "Equivalent Stress", "Fixator Mass"))

        fig.add_trace(
                go.Scatter(x =  y_test[:,0], y = y_pred_test_1, mode = 'markers',marker=dict(color='blue',size=8)),
                row=1, col=1)

        fig.add_trace(
                go.Scatter(x =  [5,12], y = [5,12], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
                row=1, col=1)

        fig.update_xaxes(title_text="Simulation", range=[5,12],showgrid=False, row=1, col=1)
        fig.update_yaxes(title_text="Predicted", range=[5,12], showgrid=False, row=1, col=1)


        fig.add_trace(
                go.Scatter(x =  y_test[:,1], y = y_pred_test_2, mode = 'markers',marker=dict(color='blue',size=8)),
                row=1, col=2)

        fig.add_trace(
                go.Scatter(x =  [200,550], y = [200,550], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
                row=1, col=2)


        fig.update_xaxes(title_text="Simulation", range=[200,550],showgrid=False, row=1, col=2)
        fig.update_yaxes(range=[200,550], showgrid=False, row=1, col=2)

        fig.add_trace(
                go.Scatter(x =  y_test[:,2], y = y_pred_test_3, mode = 'markers',marker=dict(color='blue',size=8)),
                row=1, col=3)

        fig.add_trace(
                go.Scatter(x =  [0.2,0.32], y = [0.2,0.32], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
                row=1, col=3)


        fig.update_xaxes(title_text="Simulation", range=[0.2,0.32],showgrid=False, row=1, col=3)
        fig.update_yaxes(range=[0.2,0.32], showgrid=False, row=1, col=3)

        fig.update_layout(height=400, width=1000, showlegend=False)
        st.write(fig)
        
        with st.expander("Error table"):

        
            st.dataframe(df_val_error)
        
        
        with st.expander("Design Parameters Sensitivity"):
            st.write("The importance score for each input parameter")
        #st.markdown("<hr/>", unsafe_allow_html=True) 

    
            fig_col1, fig_col2, fig_col3  = st.columns([0.5,3,1])  

            with fig_col2:
                feature_importances = pd.DataFrame(model2.feature_importances_,index = df6.columns[0:6],columns=['importance']).sort_values('importance', ascending=False)
                num = feature_importances.shape[0]
                ylocs = np.linspace(1,num,num)
                values_to_plot = feature_importances[:num].values.ravel()[::-1]
                feature_labels = list(feature_importances[:num].index)[::-1]
                #plt.figure(num=None, facecolor='w', edgecolor='k');
                plt.barh(ylocs, values_to_plot, align = 'center', height = 0.8)
                plt.ylabel('Features')
                plt.xlabel('Featur importance score')
                plt.yticks(ylocs, feature_labels)
                st.pyplot(plt)

#----------------------------------------------------------------------------------------------  

    if 'result' in st.session_state:
        st.sidebar.success('Taining is done !')
        val_error()


    if 'result' not in st.session_state:    
        result = st.sidebar.button("Start training")

        if result:
    
            st.session_state.result = 1

            st.spinner()
            with st.spinner('Training AI model...'):
    
                time.sleep(5)
                st.balloons()

                st.sidebar.success('Taining is done !')
        
                val_error()
    


elif option=="Performance Prediction":

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
        