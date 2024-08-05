import streamlit as st
import re
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Download and load the model and tokenizer from Hugging Face
try:
    model_path = hf_hub_download(repo_id="Yaaba/Final_Project", filename="fasttext_model")
   

    # Load the model and tokenizer
    model = AutoModel.from_pretrained(model_path)
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Define functions for text preprocessing and prediction
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def predict_next_word(model, tokenizer, context, top_n=7):
    # Preprocess the context
    context = preprocess_text(context)
    inputs = tokenizer(context, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits[0, -1].detach().numpy()

    # Get top_n predictions
    top_indices = predictions.argsort()[-top_n:][::-1]
    next_words = [tokenizer.decode(idx) for idx in top_indices]

    return next_words

def generate_story(model, tokenizer, start_prompt, max_words=100):
    story = start_prompt.split()
    recent_words = []

    for _ in range(max_words):
        context = ' '.join(story[-3:])  # Use the last 3 words as context
        next_word_predictions = predict_next_word(model, tokenizer, context, top_n=10)

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
    generated_story = generate_story(model, tokenizer, start_prompt, max_words)
    st.write("Generated Story:")
    st.write(generated_story)

# Input context for next word prediction
st.subheader('Predict Next Words')
context = st.text_input('Context', 'me pɛ')
top_n = st.slider('Top N Predictions', min_value=1, max_value=10, value=5)
if st.button('Predict Next Words'):
    next_words = predict_next_word(model, tokenizer, context, top_n)
    st.write("Next Word Predictions:")
    st.write(next_words)
