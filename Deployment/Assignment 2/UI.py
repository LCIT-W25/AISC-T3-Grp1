import streamlit as st
import numpy as np
from Processing import Processing
from Predect import Predect
from pickle_1 import Pickle
from Joblib import Joblib
from Interpertable import Interpertable
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    processing = Processing()
    predect = Predect()
    pickle = Pickle()
    joblib = Joblib()
    interpertable = Interpertable()

    model_path = "./Models"

    rnn = "RNN (Recurrent Neural Network)"
    gnn = "GNN (Graph Neural Network)"

    img_model_file_name = {
        "CNN (Convolutional Neural Network)" : "cnn_photo",
        "EfficientNet (Transfer Learning Model)" : "EfficientNetB0",
    }
    text_model_file_name =  {
        rnn : "rnn_model",
        gnn : "gnn_model",
    }

    st.set_page_config(page_title="Text Classification App", layout="wide")

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Text Classification", "Image Classification"))

    if page == "Text Classification":
        st.title("Text Classification - Multiple Models")
        label = {0: 'Strong_Pos', 1: 'Mild_Pos', 2: "Neutral", 3: 'Strong_Neg', 4: 'Mild_Neg'}

        st.subheader("Enter Text & Predict")
        # A text area (multi-line) for user input
        user_input = st.text_area("Your text here:", height=100)

        # Create a dropdown for 5 models
        model_choice = st.selectbox(
            "Select a Model:",
            [
                rnn,
                gnn
            ]
        )

        # Disable Predict button if text input is empty
        predict_disabled = not user_input.strip()

        # A "Predict" button (enabled only if text input is provided)
        if st.button("Predict", disabled=predict_disabled):
            # Preprocess the input text
            processed_data = processing.test_precessing(user_input)

            if model_choice == rnn:
                # Load the tokenizer and transform text input
                tokenizer = joblib.load_model_from_specified_path(model_path, "rnn_tokenizer")
                
                # Tokenize and pad the input text
                sequence_data = tokenizer.texts_to_sequences(processed_data)
                padded_data = pad_sequences(sequence_data, maxlen=50, padding='post', truncating='post')  # Adjust maxlen as needed
                
                # Load RNN model and make prediction
                rnn_model = joblib.load_model_from_specified_path(model_path, "rnn_model")
                prediction_text = predect.predict_target(rnn_model, padded_data)

            elif model_choice == gnn:
                # Load Word2Vec model for GNN
                word2vec_model = joblib.load_model_from_specified_path(model_path, "gnn_word2vec_model")
                
                # Convert the processed tweet into a vector using Word2Vec
                vectorized_input = processing.get_tweet_vector(processed_data, word2vec_model)
                vectorized_input = np.array([vectorized_input])

                reshaped_input = vectorized_input.reshape((vectorized_input.shape[0], 1, vectorized_input.shape[1]))  # Reshaping for GRU

                # Predict based on the vectorized input
                model = joblib.load_model_from_specified_path(model_path, text_model_file_name[model_choice])
                prediction_text = predect.predict_target(model, reshaped_input)  # Assuming predict_target can handle vectors


            # Show the result below the button
            st.subheader("Model Prediction Result")
            # Get the class with the highest probability
            predicted_class_index = np.argmax(prediction_text[0])  # Get index of class with highest probability
            predicted_class = label[predicted_class_index]  # Map index to class name

            # Display the predicted class
            st.write(f"**Model Prediction**: <span style='color: green'>{predicted_class}</span>", unsafe_allow_html=True)


            # Assuming `prediction_text` contains the probability distribution across all classes (e.g., in the form of a softmax output)

            # Get the probabilities for each class
            class_probs = prediction_text[0]  # Assuming prediction_text is an array of class probabilities

            # Show each class and its probability
            st.subheader("Class Probabilities")
            for i, prob in enumerate(class_probs):
                st.write(f"**Class**: {label[i]} - **Probability**: {prob * 100:.2f}%")


    elif page == "Image Classification":
        st.title("Image Classification - Multiple Models")
        label =  {0 : 'food',1 : 'inside',2 : 'outside',3 : 'drink',4 : 'menu'}

        st.subheader("Upload Image & Predict")
        uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)
            st.success("File uploaded successfully!")

        # Create a dropdown for 5 models
        model_choice = st.selectbox(
            "Select a Model:",
            [
                "CNN (Convolutional Neural Network)",
                "EfficientNet (Transfer Learning Model)"
            ]
        )

        # Disable Predict button if no file is uploaded
        predict_disabled = uploaded_file is None

        # A "Predict" button (enabled only if a file is uploaded)
        if st.button("Predict", disabled=predict_disabled):
            # For now, just display a placeholder until the model endpoints are ready.

            processed_data = processing.preprocess_image(uploaded_file)

            array = np.expand_dims(processed_data, axis=0)

            model = joblib.load_model_from_specified_path(model_path, img_model_file_name[model_choice])
                
            prediction_text = predect.predict_target(model, array)

            predicted_class = np.argmax(prediction_text, axis=1)[0]

            st.subheader("Model Prediction Result")
            st.write(f"**Predicted Class**: <span style='color: green'>{label[predicted_class]}</span>", unsafe_allow_html=True)

            # Display the probabilities of all classes
            st.subheader("Prediction Probabilities for All Classes")
            class_probs = prediction_text[0]  # Assuming prediction_probs is of shape (1, num_classes)
            
            # Show each class and its probability
            for i, prob in enumerate(class_probs):
                st.write(f"Class: **{label[i]}** - Probability: {prob * 100:.2f}%")

            # Optionally, generate LIME explanation with the grayed-out background
            lime_img = interpertable.explain_with_lime_for_image(model, array)

            # Display the LIME explanation
            st.subheader("LIME Explanation")
            st.image(lime_img, caption="LIME Explanation", use_column_width=False, width=300)

if __name__ == "__main__":
    main()