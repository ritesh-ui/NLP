#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import spacy
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC
global nlp
nlp=spacy.load('en_core_web_sm')
global df
stemmer=SnowballStemmer(language='english')


# In[2]:


def understanding_data():
    print("First 5 records")
    print(df.head())
    print('\n')
    print("Information of the data")
    print(df.info())
    print('\n')
    print("Shape of the data")
    print(df.shape)
    print('\n')
    print(df['label'].unique())
    print('\n')
    print(df.label.value_counts(normalize=True))
    print('\n')
    


# ### Data Preprocessing

# In[3]:


def remove_stopwords(tokens):
    return [token.text for token in tokens if not token.is_stop] 


# In[4]:


def lemmatization(tokens):
    return [token.lemma for token in tokens]


# In[141]:


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
    


# In[67]:


def test_train_split(df):
    #Dropping unwanted columns
    df.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    #Changing column into different name
    df['label']=np.where(df['label']=='REAL',1,0)
    X=df.drop(['label'],axis=1)
    Y=df.pop('label')
    X['title']=[" ".join(X['title'][i]) for i in range(len(X))]
    X['text']=[" ".join(X['text'][i]) for i in range(len(X))]
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
    return (x_train,x_test,y_train,y_test)


# ### Creating Wordcloud

# In[68]:


def show_wordcloud(data, title = None):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# ### Creating Transformers

# In[125]:


def transformers(df):
    transformer = FeatureUnion([
                ('search_term_tfidf', 
                  Pipeline([('extract_field',
                              FunctionTransformer(lambda x: x['title'], 
                                                  validate=False)),
                            ('tfidf', 
                              TfidfVectorizer())])),
                ('product_title_tfidf', 
                  Pipeline([('extract_field', 
                              FunctionTransformer(lambda x: x['text'], 
                                                  validate=False)),
                            ('tfidf', 
                              TfidfVectorizer())]))])
    
    return transformer 


def model_making_2(x):
    transformer=transformers(x)
    model = Pipeline([('transformer',transformer),('svc',SVC(C=1.0,
    kernel='sigmoid',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=True,
    tol=0.001,
    cache_size=200,
    class_weight=None,
    verbose=False,
    max_iter=-1,
    decision_function_shape='ovr',
    break_ties=False,
    random_state=None))])
    return model



