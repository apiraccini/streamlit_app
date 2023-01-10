# prototype ml app, with pycaret and pandas profiling

import streamlit as st
import pandas as pd

from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True)['frame']
train = df.sample(frac=0.75, random_state=42)
test = df.drop(train.index)

with st.sidebar: 
    st.title('Simple autoML app')
    st.image('icon.png')
    choice = st.radio('Navigation', ['Profiling', 'Modelling', 'Download'])
    st.info('This project application trains and build a model on the California Housing Dataset.')

if choice == 'Profiling': 
    st.title('Exploratory Data Analysis')
    profile_df = train.profile_report()
    st_profile_report(profile_df)

if choice == 'Modelling': 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):

        # set up pycaret
        setup(train, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)

        # compare many models and save best
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == 'Download': 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name='best_model.pkl')