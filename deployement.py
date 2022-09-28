# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:12:46 2022

@author: haari
"""

import pandas as pd
import numpy as np
import docx2txt
import streamlit as st
import pdfplumber
import re
import nltk 
from nltk.tokenize import RegexpTokenizer 
from nltk import word_tokenize
from nltk.stem import  WordNetLemmatizer 
from wordcloud import WordCloud 
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
import matplotlib.pyplot as plt
from pickle import load
import pickle

model=load(open(r"D:\Project_2\clf_xgb.pickle",'rb'))
vectors=load(open(r"D:\Project_2\tfidf.pickle",'rb'))



resume=[]

def display(doc_file):
    if doc_file.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
    return resume
            

def preprocess(sentence):
    sentence1=str(sentence)
    sentence2=sentence1.lower()
    sentence3=sentence2.replace('{html}',"")
    cleanr=re.compile('<.*?>')
    cleantext=re.sub(cleanr,'',sentence3)
    rem_url=re.sub(r'http\S+','',cleantext)
    rem_num=re.sub('[0-9]+','',rem_url)
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(rem_num)
    filtered_words=[w for w in tokens if len(w)>2 if not w in stopwords.words('english')]
    lemmatizer=WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return" ".join(lemma_words)
    
def main():
    st.title("Resume Classifier")
    menu=["Home","Resume Classifier",'About']
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice=="Home":
        st.header("Welcome to Resume Classifier")
        st.markdown("![Alt Text](https://affinda.com/wp-content/uploads/2020/07/animation_640_kcd9gke3-1.gif)")
    
    elif choice=="Resume Classifier":
        upload_file=st.file_uploader('Upload a resume file here', type=['docx','pdf'], accept_multiple_files=True)
        
        if st.button('Predict'):
            for doc_file in upload_file:
                if doc_file is not None:
                
                    file_details={'filename':[doc_file.name],
                              'filetype':doc_file.type.split('.')[-1].upper(),
                              'filesize':str(doc_file.size)+' KB'}
                
                    file_type=pd.DataFrame(file_details)
                    st.write(file_type.set_index('filename'))
                    displayed=display(doc_file)
                    cleaned=preprocess(displayed)
                    predicted=model.predict(vectors.transform([cleaned]))
                
                    if int(predicted)==0:
                        st.success("This resume belongs to PeopleSoft Resumes")
                    elif int(predicted)==1:
                        st.success("This resume belongs to ReactJS Developer")
                    elif int(predicted)==2:
                        st.success("This resume blongs to SQL Developer Lightning Insights")
                    else:
                        st.success("This Resume belongs to Workday Resume")
                        
    elif choice=="About":
         st.header("This project is made to significantly reduce the effort in HRM department. This project is made under guidance of R P Adhvaith Sir  and Deepika Mam. ")
         st.subheader("Mohammed Haaris")
         st.subheader("Mohd. Javed Ansari")
         st.subheader("Sumeet Yogeshkumar Vaidya")
         st.subheader("Shaik Hafeez")
         st.subheader("Anjum Anwar Shaik")
         st.subheader("Subendhu Mishra")
    
    
    
    
if __name__ == '__main__':
	main()   
    