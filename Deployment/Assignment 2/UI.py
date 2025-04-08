import streamlit as st
import numpy as np
import pandas as pd
from Processing import Processing
from Predict import Predict  # Renamed from 'Predect'
from pickle_1 import Pickle
from Joblib import Joblib
from Interpertable import Interpertable
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    processing = Processing()
    predict = Predict()
    pickle = Pickle()
    joblib = Joblib()
    interpertable = Interpertable()

    model_path = "./Models"

    rnn = "RNN (Recurrent Neural Network)"
    gnn = "GNN (Graph Neural Network)"

    text_model_file_name = {
        rnn: "rnn_model",
        gnn: "gnn_model",
    }

    st.set_page_config(page_title="Text Classification App", layout="wide")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Text Classification", "Image Classification"))

    if page == "Text Classification":
        st.title("Text Classification - Multiple Models")
        label = {0: 'Strong_Pos', 1: 'Mild_Pos', 2: "Neutral", 3: 'Strong_Neg', 4: 'Mild_Neg'}

        st.subheader("Enter Text & Predict")
        user_input = st.text_area("Your text here:", height=100)

        model_choice = st.selectbox(
            "Select a Model:",
            [rnn, gnn]
        )

        predict_disabled = not user_input.strip()

        if st.button("Predict", disabled=predict_disabled):
            try:
                processed_data = processing.test_precessing(user_input)

                if model_choice == rnn:
                    # Load tokenizer
                    tokenizer = joblib.load_model_from_specified_path(model_path, "rnn_tokenizer")
                    if not tokenizer:
                        st.error("Tokenizer loading failed.")
                        return
                    
                    sequence_data = tokenizer.texts_to_sequences(processed_data)
                    padded_data = pad_sequences(sequence_data, maxlen=50, padding='post', truncating='post')

                    rnn_model = joblib.load_model_from_specified_path(model_path, "rnn_model")
                    if not rnn_model:
                        st.error("RNN model loading failed.")
                        return

                    prediction_text = predict.predict_target(rnn_model, padded_data)

                elif model_choice == gnn:
                    word2vec_model = joblib.load_model_from_specified_path(model_path, "gnn_word2vec_model")
                    if not word2vec_model:
                        st.error("Word2Vec model loading failed.")
                        return

                    vectorized_input = processing.get_tweet_vector(processed_data, word2vec_model)
                    vectorized_input = np.array([vectorized_input])
                    reshaped_input = vectorized_input.reshape((vectorized_input.shape[0], 1, vectorized_input.shape[1]))

                    gnn_model = joblib.load_model_from_specified_path(model_path, "gnn_model")
                    if not gnn_model:
                        st.error("GNN model loading failed.")
                        return

                    prediction_text = predict.predict_target(gnn_model, reshaped_input)

                if prediction_text is not None and len(prediction_text[0]) == len(label):
                    predicted_class_index = np.argmax(prediction_text[0])
                    predicted_class = label[predicted_class_index]

                    st.subheader("Model Prediction Result")
                    st.markdown(f"**Model Prediction**: <span style='color: green'>{predicted_class}</span>", unsafe_allow_html=True)

                    # Optional: Show confidence bar chart
                    df = pd.DataFrame({
                        "Label": list(label.values()),
                        "Confidence": prediction_text[0]
                    })
                    st.bar_chart(df.set_index("Label"))
                else:
                    st.error("Unexpected prediction output format.")

            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
