import streamlit as st
from Fake_News_Classification import data_preprocessing,remove_stopwords,lemmatization
import numpy as np
import pandas as pd
import dill
import pandas as pd
import numpy as np
import spacy
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer(language='english')
st.write("""
# Fake News Classifier
""")
st.write("""
***Enter News Title and Story to check its authenticity*** 
""")

add_selectbox = st.sidebar.selectbox(
    'Do you want to tweak threshold?',
    ('Yes', 'No')
)

if add_selectbox == 'Yes':
    st.sidebar.write("")
    st.sidebar.write("""***Enter your probability threshold ***""")
    threshold=st.sidebar.slider("",0.0, 1.0, (0.50))
#st.sidebar.selectbox("Do you want to tweak threshold?")


st.subheader('News Title')
title = st.text_input(label="")

st.subheader('Enter News Snippet')
news  = st.text_input(" ")

def prediction(df):
    with open('nlp.pkl', 'rb') as file:
        prediction = dill.load(file)
    
    if add_selectbox =='Yes':
        if prediction.predict_proba(df)[0,0] > threshold:
           st.write(""" 
            # News is Fake
            """)
        else:       
            st.write(""" 
            # News is True
            """)
        st.subheader('Probability of News being fake')
        st.write(prediction.predict_proba(df)[0,0])
        
    else:
        if prediction.predict(df) == 0:
            st.write(""" 
            # News is Fake
            """)
        else:
            st.write(""" 
            # News is True
            """)
        
        st.subheader('Probability of News being fake')
        st.write(prediction.predict_proba(df)[0,0])

load_button = st.button('Submit')

if load_button:
    dictionary={'title':title,'text':news}
    df=pd.DataFrame(dictionary,index=[0])
    df=data_preprocessing(df)
    df['title']=[" ".join(df['title'][i]) for i in range(len(df))]
    df['text']=[" ".join(df['text'][i]) for i in range(len(df))]
    prediction(df)
    
 







