import streamlit as st
import numpy as np
import pandas as pd
from Processing import Processing
from Predict import Predict
from pickle_1 import Pickle
from Joblib import Joblib
from Interpertable import Interpertable
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
LABELS = {0: 'Strong_Pos', 1: 'Mild_Pos', 2: "Neutral", 3: 'Strong_Neg', 4: 'Mild_Neg'}
MODEL_PATH = "./Models"
RNN = "RNN (Recurrent Neural Network)"
GNN = "GNN (Graph Neural Network)"


def load_tokenizer_and_pad(user_input, joblib, model_path):
    tokenizer = joblib.load_model_from_specified_path(model_path, "rnn_tokenizer")
    if not tokenizer:
        return None, "Tokenizer loading failed."

    processed_data = Processing().test_precessing(user_input)
    sequence_data = tokenizer.texts_to_sequences(processed_data)
    padded_data = pad_sequences(sequence_data, maxlen=50, padding='post', truncating='post')
    return padded_data, None


def handle_rnn_prediction(user_input, joblib, predict):
    padded_data, error = load_tokenizer_and_pad(user_input, joblib, MODEL_PATH)
    if error:
        return None, error

    rnn_model = joblib.load_model_from_specified_path(MODEL_PATH, "rnn_model")
    if not rnn_model:
        return None, "RNN model loading failed."

    prediction = predict.predict_target(rnn_model, padded_data)
    return prediction, None


def handle_gnn_prediction(user_input, joblib, predict):
    processing = Processing()
    word2vec_model = joblib.load_model_from_specified_path(MODEL_PATH, "gnn_word2vec_model")
    if not word2vec_model:
        return None, "Word2Vec model loading failed."

    processed_data = processing.test_precessing(user_input)
    vectorized_input = processing.get_tweet_vector(processed_data, word2vec_model)
    reshaped_input = np.array([vectorized_input]).reshape((1, 1, -1))

    gnn_model = joblib.load_model_from_specified_path(MODEL_PATH, "gnn_model")
    if not gnn_model:
        return None, "GNN model loading failed."

    prediction = predict.predict_target(gnn_model, reshaped_input)
    return prediction, None


def display_prediction_result(prediction_text):
    if prediction_text is not None and len(prediction_text[0]) == len(LABELS):
        predicted_class_index = np.argmax(prediction_text[0])
        predicted_class = LABELS[predicted_class_index]

        st.subheader("Model Prediction Result")
        st.markdown(f"**Model Prediction**: <span style='color: green'>{predicted_class}</span>", unsafe_allow_html=True)

        df = pd.DataFrame({
            "Label": list(LABELS.values()),
            "Confidence": prediction_text[0]
        })
        st.bar_chart(df.set_index("Label"))
    else:
        st.error("Unexpected prediction output format.")


def text_classification_ui():
    st.title("Text Classification - Multiple Models")
    st.subheader("Enter Text & Predict")
    user_input = st.text_area("Your text here:", height=100)

    model_choice = st.selectbox("Select a Model:", [RNN, GNN])
    predict_disabled = not user_input.strip()

    if st.button("Predict", disabled=predict_disabled):
        joblib_obj = Joblib()
        predict_obj = Predict()

        if model_choice == RNN:
            prediction, error = handle_rnn_prediction(user_input, joblib_obj, predict_obj)
        else:
            prediction, error = handle_gnn_prediction(user_input, joblib_obj, predict_obj)

        if error:
            st.error(error)
        else:
            display_prediction_result(prediction)


def main():
    st.set_page_config(page_title="Text Classification App", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Text Classification", "Image Classification"))

    if page == "Text Classification":
        text_classification_ui()


if __name__ == "__main__":
    main()
