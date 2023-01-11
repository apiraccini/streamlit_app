# prototype ml app, with pycaret and pandas profiling

import streamlit as st
import pandas as pd

from pycaret.regression import *
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True)['frame']
train = df.sample(frac=0.75, random_state=42)
test = df.drop(train.index)

with st.sidebar: 
    st.title('Simple autoML app')
    st.image('icon.png')
    choice = st.radio('Navigation', ['Exploratory Data Analysis', 'Machine learning', 'Download best model'])
    st.info('This project application trains and build a model on the California Housing Dataset.')

if choice == 'Exploratory Data Analysis': 
    st.title(choice)
    profile_df = train.profile_report()
    st_profile_report(profile_df)

if choice == 'Machine learning':
    st.title(choice) 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)

    # set up pycaret
    setup(train, target=chosen_target)
    setup_df = pull()
    st.dataframe(setup_df)
    
    if st.button('Compare different models'):
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        
if choice == 'Download best model':
    st.title(choice) 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name='best_model.pkl')