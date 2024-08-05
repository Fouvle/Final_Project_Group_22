# Final_Project_Group_22
Deployment Link: https://huggingface.co/spaces/Yaaba/fasttext
Youtube LINK: https://youtu.be/yWJLrzF9L9I

Certainly! Here's a README generated from the provided code:

# Twi Story Generator

This project implements a Twi language story generator using Word2Vec and FastText models. It includes data preprocessing, model training, optimization, and evaluation. The project also features a Streamlit web application for generating stories.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [File Structure](#file-structure)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Streamlit Application](#streamlit-application)

## Installation

To run this project, you need to install the following dependencies:

```
pip install gensim pandas numpy tensorflow sklearn transformers torch torchvision torchaudio streamlit
```

## Usage

1. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Load and preprocess the datasets:
   ```python
   test_text = pd.read_csv('/content/drive/MyDrive/Project/test.txt', delimiter='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
   validation_text = pd.read_csv('/content/drive/MyDrive/Project/dev.txt', delimiter='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
   train_text = pd.read_csv('/content/drive/MyDrive/Project/train.txt', delimiter='\t', encoding='utf-8', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
   ```

3. Train and optimize the models:
   ```python
   # Word2Vec
   optimized_model = optimize_word2vec(train_tokenized)
   optimized_model.save("optimized_word2vec_model.model")

   # FastText
   optimized_model = optimize_fasttext(train_tokenized)
   ```

4. Evaluate the models:
   ```python
   accuracy = evaluate_word2vec_model(model, validation_tokenized, top_n=5)
   print(f"Validation Accuracy: {accuracy}")
   ```

5. Generate stories:
   ```python
   generated_story = generate_story(optimized_model, start_prompt, max_words=5)
   print("Generated Story:", generated_story)
   ```

## Features

- Text preprocessing and tokenization
- Word2Vec and FastText model training and optimization
- Next word prediction
- Model evaluation
- Story generation
- Streamlit web application for interactive story generation

## File Structure

- `project.Word2VEC_Predictionipynb`: Main project file
- `word2vec_model.model`: Saved Word2Vec model
- `fasttext_model.model`: Saved FastText model
- `word_embeddings.kv`: Saved word embeddings
- `optimized_word_embeddings.txt`: Optimized word embeddings in text format
- `fasttext_model.pkl`: Saved optimized FastText model

## Model Training

The project trains both Word2Vec and FastText models on the Twi language dataset. The models are then optimized for better performance.

## Evaluation

The models are evaluated using custom functions that measure the accuracy of next word predictions on validation and test datasets.

## Streamlit Application

The project includes a Streamlit web application that allows users to generate Twi stories interactively. Users can input a starting prompt and specify the number of words to generate.

To run the Streamlit app:

```
streamlit run app.py
```

Replace `app.py` with the name of your Streamlit script.

---

This README provides an overview of the project, its features, and instructions for usage. You may want to expand on certain sections or add more details specific to your implementation.

For The Deployment
Here's a README based on the provided code:

# FastText Text Generation and Prediction

This Streamlit application demonstrates text generation and next word prediction using a FastText model trained on Twi language data.

## Features

1. Story Generation: Generate a story based on a given prompt.
2. Next Word Prediction: Predict the most likely next words given a context.

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   ```

2. Install the required packages:
   ```
   pip install streamlit gensim numpy
   ```

3. Ensure you have the `fasttext_model.pkl` file in the same directory as the script.

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

Replace `app.py` with the name of your script file.

## Application Structure

### Story Generation

- Input a start prompt
- Choose the maximum number of words to generate
- Click "Generate Story" to create a story based on the input

### Next Word Prediction

- Input a context (a word or phrase)
- Select the number of top predictions to display
- Click "Predict Next Words" to see the most likely next words

## Functions

1. `preprocess_text(text)`: Cleans and preprocesses input text.
2. `predict_next_word(model, context, top_n=7)`: Predicts the next words given a context.
3. `generate_story(model, start_prompt, max_words=100)`: Generates a story based on a starting prompt.

## Model

The application uses a pre-trained FastText model (`fasttext_model.pkl`) for text generation and prediction. Ensure this file is present in the same directory as the script.

## Error Handling

If the model fails to load, an error message will be displayed, and the application will stop.

## Note

This application is designed for the Twi language. Ensure that your input text and prompts are in Twi for the best results.

---

