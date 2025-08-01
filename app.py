import streamlit as st
import numpy as np
import pandas as pd
import joblib

# First lets load the instances that were created

with open('scaler.joblib','rb') as file:
    scale = joblib.load(file)
    
with open('pca.joblib','rb') as file:
    pca = joblib.load(file)
    
with open('final_model.joblib','rb') as file:
    model = joblib.load(file)
    
def prediction(input_list):
    
    scaled_input = scale.transform([input_list])
    pca_input = pca.transform(scaled_input)
    output = model.predict(pca_input)[0]
    
    if output == 0:
        return 'Developed'
    elif output == 1:
        return 'Underdeveloped'
    else:
        return 'Developing'
    
def main():
    
    st.title('HELP NGO FOUNDATION')
    st.subheader('This application will give the status of a country based on socio-economic and health factors')
    
    gdp = st.text_input('Enter the GDP per population of a country')
    inc = st.text_input('Enter the per capita income of a country')
    imp = st.text_input('Enter the Imports in terms of % of GDP')
    exp = st.text_input('Enter the Exports in terms of % of GDP')
    inf = st.text_input('Enter the inflation rate in a country(%)')
    
    hel = st.text_input('Enter the expenditure on health in terms % of GDP')
    ch_m = st.text_input('Enter the number of deaths per 1000 births for <5 years')
    fer = st.text_input('Enter the average children born to a women in a country')
    lf = st.text_input('Enter the average life expectancy in a country')
    
    in_data=[ch_m,exp,hel,imp,inc,inf,lf,fer,gdp]
    
    if st.button('Predict'):
        response = prediction(in_data)
        st.success(response)
        
if __name__=='__main__':
    main()
