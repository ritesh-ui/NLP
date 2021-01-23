import streamlit as st
#from Fake_News_Classification import data_preprocessing,remove_stopwords,lemmatization
import numpy as np
import pandas as pd
import dill
import pandas as pd
import numpy as np
import spacy
import nltk
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

def remove_stopwords(tokens):
    return [token.text for token in tokens if not token.is_stop] 

def lemmatization(tokens):
    return [token.lemma for token in tokens]

def data_preprocessing(df): #Cleans the data and return x_train ,y_train,x_test,y_test

    from nltk.stem.snowball import SnowballStemmer
    stemmer=SnowballStemmer(language='english')
    import spacy
    nlp=spacy.load('en_core_web_sm')
    #Removing punctuations
    df['title']=df['title'].str.replace('[^\w\s]','')
    df['text']=df['text'].str.replace('[^\w\s]','')
    
    #Stemming every line
    df['title']=[stemmer.stem(y) for y in df['title']]
    df['text']= [stemmer.stem(y) for y in df['text']]
    
    #Tokeninzing every words
    df['title']=df['title'].apply(lambda x:nlp.tokenizer(x))
    df['text']=df['text'].apply(lambda x:nlp.tokenizer(x))
    
    #Stopping words removal
    df['title']=df['title'].apply(remove_stopwords)
    df['text']=df['text'].apply(remove_stopwords)
    
    #Replacing few unwanted characters
    df['title'] =[list(map(lambda x:x.replace("\n",""),df['title'][i])) for i in range(len(df))]
    df['text'] =[list(map(lambda x:x.replace("\n",""),df['text'][i])) for i in range(len(df))]
    df['title'] =[list(map(lambda x:x.replace(" ",""),df['title'][i])) for i in range(len(df))]
    df['text'] =[list(map(lambda x:x.replace(" ",""),df['text'][i])) for i in range(len(df))]
    df['title'] =[list(filter(None, df['title'][i])) for i in range(len(df))]
    df['text'] =[list(filter(None, df['text'][i])) for i in range(len(df))]  
    return (df)

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
    
 







