import streamlit as st
import re
import pickle
import numpy as np

# Attempt to download and load the model from Hugging Face
try:
    model = pickle.load(open('fasttext_model.pkl', 'rb'))
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

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

    # Use the last word(s) as the context
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

# Streamlit app
st.title('FastText Text Generation and Prediction')

# Input prompt for story generation
st.subheader('Generate a Story')
start_prompt = st.text_input('Start Prompt', 'sɛ')
max_words = st.slider('Maximum Words', min_value=1, max_value=100, value=50)
if st.button('Generate Story'):
    generated_story = generate_story(model, start_prompt, max_words)
    st.write("Generated Story:")
    st.write(generated_story)

# Input context for next word prediction
st.subheader('Predict Next Words')
context = st.text_input('Context', 'me pɛ')
top_n = st.slider('Top N Predictions', min_value=1, max_value=10, value=5)
if st.button('Predict Next Words'):
    next_words = predict_next_word(model, context, top_n)
    st.write("Next Word Predictions:")
    st.write(next_words)
