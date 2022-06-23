import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import re

def lower(text):
    return text.lower()


exclude = string.punctuation


def remove_punctuation(text):
    return text.translate(str.maketrans('','',exclude))


stwords = stopwords.words('english')


def remove_stopwords(text):
    new = []
    for t in text.split():
        if t in stwords:
            new.append('')
        else:
            new.append(t)
    return " ".join(new)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("IMDB Movie Review WebApp")
st.subheader("Built by - Aryan Singh")
text = st.text_area('Enter the Review')


if st.button('Predict'):

    if len(text) == 0 :
        st.header("Please enter a valid text ")

    #lower
    lower_text = lower(text)

    #punctuation
    new_text = remove_punctuation(lower_text)

    #removestopwords
    new_text1 = remove_stopwords(new_text)

    #vectorization
    vector_input = tfidf.transform([new_text1])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Positive")

    else:
        st.header("Negative")









