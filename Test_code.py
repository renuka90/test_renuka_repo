# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:32:02 2020

@author: ICTO-EB
"""


"""
Created on Mon Feb  3 01:59:14 2020

@author: Renu
"""
import streamlit as st
import pandas as pd
from gensim.models import Word2Vec

st.title('Search tool: Journal Of Management_120 pdfs')
pos_str = st.text_input('Enter keyword(s)')



model = Word2Vec.load('save_model1_with_6_articles.model')
# TRAIN THE WORD2VEC MODEL ON THE DATA


if st.button("Search"):
    result = pd.DataFrame(model.wv.most_similar(positive = pos_str, topn=10), columns = ['word', 'similarity'])
    st.text(result)