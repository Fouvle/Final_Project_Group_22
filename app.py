# app.py
import streamlit as st
from gensim.models import FastText
import re
import pickle
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Yaaba/Final_Project", filename="fasttext_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(model.feature_names_in_)


# Define functions for text preprocessing and prediction
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def predict_next_word(model, context, top_n=7):
    # Preprocess the context
    context = preprocess_text(context)
    words = context.split()

    if not words:
        return []

    last_word = words[-1]

    if last_word not in model.wv:
        return []

    # Find the top_n most similar words
    similar_words = model.wv.most_similar(last_word, topn=top_n)

    # Extract the words from the similar words list
    next_words = [word for word, similarity in similar_words]

    return next_words

def generate_story(model, start_prompt, max_words=100):
    story = start_prompt.split()
    recent_words = []

    for _ in range(max_words):
        context = ' '.join(story[-3:])  # Use the last 3 words as context
        next_word_predictions = predict_next_word(model, context, top_n=10)

        if not next_word_predictions:
            break

        # Choose the most probable word that hasn't been used recently
        for word in next_word_predictions:
            if word not in recent_words:
                next_word = word
                break
        else:
            # If all words have been used recently, use the most probable word
            next_word = next_word_predictions[0]

        story.append(next_word)

        # Keep track of recent words
        recent_words.append(next_word)
        if len(recent_words) > 10:  # Adjust this number as needed
            recent_words.pop(0)

    return ' '.join(story)

# Load the FastText model
with open('fasttext_model.pkl', 'rb') as file:
    optimized_model = pickle.load(file)

# Streamlit app
st.title('FastText Text Generation and Prediction')

# Input prompt for story generation
st.subheader('Generate a Story')
start_prompt = st.text_input('Start Prompt', 'sɛ')
max_words = st.slider('Maximum Words', min_value=1, max_value=100, value=50)
if st.button('Generate Story'):
    generated_story = generate_story(optimized_model, start_prompt, max_words)
    st.write("Generated Story:")
    st.write(generated_story)

# Input context for next word prediction
st.subheader('Predict Next Words')
context = st.text_input('Context', 'me pɛ')
top_n = st.slider('Top N Predictions', min_value=1, max_value=10, value=5)
if st.button('Predict Next Words'):
    next_words = predict_next_word(optimized_model, context, top_n)
    st.write("Next Word Predictions:")
    st.write(next_words)
