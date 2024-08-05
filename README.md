# Final_Project_Group_22
Deployment Link: https://huggingface.co/spaces/Yaaba/fasttext
Youtube LINK: https://youtu.be/yWJLrzF9L9I


FastText Text Generation and Prediction with Streamlit
This project showcases a Streamlit application for text generation and next-word prediction using a pretrained FastText model. The app allows users to generate stories based on a given start prompt and predict the next word(s) based on a provided context.

Features
Generate Stories: Create a story based on a starting prompt and a specified maximum number of words.
Predict Next Words: Predict the next word(s) based on the provided context.
Requirements
Streamlit
NumPy
scikit-learn (for the pretrained model)
Hugging Face FastText model
Installation
Clone the repository:

sh
Copy code
git clone <repository_url>
cd <repository_directory>
Install the required packages:

sh
Copy code
pip install streamlit numpy scikit-learn
Download the pretrained FastText model from Hugging Face and save it as fasttext_model.pkl.

Usage
Run the Streamlit app:

sh
Copy code
streamlit run app.py
Code Overview
Import Necessary Libraries
python
Copy code
import streamlit as st
import re
import pickle
import numpy as np
Load the Pretrained Model
Attempt to download and load the model from Hugging Face:

python
Copy code
try:
    model = pickle.load(open('fasttext_model.pkl', 'rb'))
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()
Define Functions for Text Preprocessing and Prediction
Preprocess Text
Remove punctuation, convert to lowercase, and strip whitespace:

python
Copy code
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()
Predict Next Word
Use the last word of the context to predict the next word(s):

python
Copy code
def predict_next_word(model, context, top_n=7):
    context = preprocess_text(context)
    words = context.split()

    if not words:
        return []

    last_word = words[-1]

    if last_word not in model.wv:
        return []

    similar_words = model.wv.most_similar(last_word, topn=top_n)
    next_words = [word for word, similarity in similar_words]

    return next_words
Generate Story
Generate a story using the given start prompt:

python
Copy code
def generate_story(model, start_prompt, max_words=100):
    story = start_prompt.split()
    recent_words = []

    for _ in range(max_words):
        context = ' '.join(story[-3:])
        next_word_predictions = predict_next_word(model, context, top_n=10)

        if not next_word_predictions:
            break

        for word in next_word_predictions:
            if word not in recent_words:
                next_word = word
                break
        else:
            next_word = next_word_predictions[0]

        story.append(next_word)
        recent_words.append(next_word)
        if len(recent_words) > 10:
            recent_words.pop(0)

    return ' '.join(story)
Streamlit App
App Title
python
Copy code
st.title('FastText Text Generation and Prediction')
Generate Story Section
python
Copy code
st.subheader('Generate a Story')
start_prompt = st.text_input('Start Prompt', 'sɛ')
max_words = st.slider('Maximum Words', min_value=1, max_value=100, value=50)
if st.button('Generate Story'):
    generated_story = generate_story(model, start_prompt, max_words)
    st.write("Generated Story:")
    st.write(generated_story)
Predict Next Words Section
python
Copy code
st.subheader('Predict Next Words')
context = st.text_input('Context', 'me pɛ')
top_n = st.slider('Top N Predictions', min_value=1, max_value=10, value=5)
if st.button('Predict Next Words'):
    next_words = predict_next_word(model, context, top_n)
    st.write("Next Word Predictions:")
    st.write(next_words)
Conclusion
This Streamlit application leverages a pretrained FastText model to provide interactive text generation and next-word prediction features. Users can input a starting prompt to generate stories or provide a context to predict the next word(s) in a sequence.

Twi Story Generator using Word2Vec and FastText
This project is a comprehensive pipeline for text preprocessing, training, evaluating, and deploying word embedding models (Word2Vec and FastText) for next-word prediction and story generation in Twi. The final output is a Streamlit web application that allows users to generate stories based on a given starting prompt.

Features
Preprocessing: Clean and tokenize text data.
Model Training: Train Word2Vec and FastText models.
Prediction: Predict the next word based on a given context.
Evaluation: Evaluate model accuracy on validation and test datasets.
Optimization: Optimize the parameters of the Word2Vec and FastText models.
Story Generation: Generate stories based on a starting prompt.
Streamlit App: Interactive web application for story generation.
Requirements
Google Colab
Python 3.x
Pandas
NumPy
TensorFlow
scikit-learn
Gensim
Streamlit
Installation
Clone the repository:

sh
Copy code
git clone <repository_url>
cd <repository_directory>
Install the required packages:

sh
Copy code
pip install pandas numpy tensorflow scikit-learn gensim streamlit
Usage
Preprocess the Text
python
Copy code
import re

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()
Load and Preprocess the Datasets
python
Copy code
import pandas as pd
import csv

# Load the datasets
test_text = pd.read_csv('/content/drive/MyDrive/Project/test.txt', delimiter='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
validation_text = pd.read_csv('/content/drive/MyDrive/Project/dev.txt', delimiter='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
train_text = pd.read_csv('/content/drive/MyDrive/Project/train.txt', delimiter='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')

# Apply preprocessing and tokenization to all datasets
train_tokenized = train_text.iloc[:, 0].apply(lambda x: preprocess_text(x).split()).tolist()
validation_tokenized = validation_text.iloc[:, 0].apply(lambda x: preprocess_text(x).split()).tolist()
test_tokenized = test_text.iloc[:, 0].apply(lambda x: preprocess_text(x).split()).tolist()
Train the Word2Vec Model
python
Copy code
from gensim.models import Word2Vec

# Initialize and train the Word2Vec model
model = Word2Vec(sentences=train_tokenized, vector_size=100, window=5, min_count=1, workers=4)

# Save the model
model.save("word2vec_model.model")

# Load the trained model
model = Word2Vec.load("word2vec_model.model")
Predict Next Word
python
Copy code
def predict_next_word(model, context, top_n=5):
    context = preprocess_text(context)
    words = context.split()
    if not words:
        return []
    last_word = words[-1]
    if last_word not in model.wv:
        return []
    similar_words = model.wv.most_similar(last_word, topn=top_n)
    next_words = [word for word, similarity in similar_words]
    return next_words

context = "me pɛ"
next_words = predict_next_word(model, context)
print("Next word predictions:", next_words)
Optimize the Word2Vec Model
python
Copy code
def optimize_word2vec(sentences, vector_size=200, window=10, min_count=1, workers=4, sg=0, hs=4, negative=0):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs, negative=negative)
    return model

optimized_model = optimize_word2vec(train_tokenized)
optimized_model.save("optimized_word2vec_model.model")
Train the FastText Model
python
Copy code
from gensim.models import FastText

# Train the FastText model
model = FastText(sentences=train_tokenized, vector_size=100, window=5, min_count=1, workers=4, sg=0, hs=5, negative=0)

# Save the model
model.save("fasttext_model.model")

# Load the trained model
model = FastText.load("fasttext_model.model")
Streamlit Application
python
Copy code
import streamlit as st

# Define functions for text preprocessing and prediction
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def predict_next_word(model, context, top_n=7):
    context = preprocess_text(context)
    words = context.split()
    if not words:
        return []
    last_word = words[-1]
    if last_word not in model.wv:
        return []
    similar_words = model.wv.most_similar(last_word, topn=top_n)
    next_words = [word for word, similarity in similar_words]
    return next_words

def generate_story(model, start_prompt, max_words=100):
    story = start_prompt.split()
    recent_words = []
    for _ in range(max_words):
        context = ' '.join(story[-3:])
        next_word_predictions = predict_next_word(model, context, top_n=10)
        if not next_word_predictions:
            break
        for word in next_word_predictions:
            if word not in recent_words:
                next_word = word
                break
        else:
            next_word = next_word_predictions[0]
        story.append(next_word)
        recent_words.append(next_word)
        if len(recent_words) > 10:
            recent_words.pop(0)
    return ' '.join(story)

st.title("Twi Story Generator")

start_prompt = st.text_input("Enter the starting text:", "sɛ")
max_words = st.slider("Number of words to generate:", min_value=10, max_value=100, value=20)

if st.button("Generate Story"):
    generated_story = generate_story(optimized_model, start_prompt, max_words)
    st.write("Generated Story:")
    st.write(generated_story)
Conclusion
This project demonstrates how to preprocess text data, train word embedding models (Word2Vec and FastText), predict next words, and generate stories. The final output is a Streamlit web application that generates stories based on a given starting prompt.














