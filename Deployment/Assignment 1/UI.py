import streamlit as st
import numpy as np
from Processing import Processing
from Predect import Predect
from pickle_1 import Pickle
from Joblib import Joblib

processing = Processing()
predect = Predect()
pickle = Pickle()
joblib = Joblib()

model_path = "./models"
label =  {0 : 'food',1 : 'inside',2 : 'outside',3 : 'drink',4 : 'menu'}

cnn = "CNN (Convolutional Neural Network)"
dnn = "DNN (Deep Neural Network)"
svm_tf_idf = "SVM (TF-IDF Vector)"

img_model_file_name = {
    cnn : "cnn_photo",
    dnn : "dnn_photo",
    "KNN (K-Nearest Neighbors)" : "knn_photo",
    "SVM (Support Vector Machine)" : "svm_photo"
}
text_model_file_name =  {
    "Naive Bayes (Count Vector)" : "nb_cv",
    "Naive Bayes (TF-IDF Vector)" : "nb_tf",
    "Naive Bayes (One-Hot Vector)" : "nb_oh",
    svm_tf_idf : "svm_tf",
    "SVM (Count Vector)" : "svm_cv"
}

st.set_page_config(page_title="Text Classification App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Text Classification", "Image Classification"))

if page == "Text Classification":
    st.title("Text Classification - Multiple Models")

    st.subheader("Enter Text & Predict")
    # A text area (multi-line) for user input
    user_input = st.text_area("Your text here:", height=100)

    # Create a dropdown for 5 models
    model_choice = st.selectbox(
        "Select a Model:",
        [
            "Naive Bayes (Count Vector)",
            "Naive Bayes (TF-IDF Vector)",
            "Naive Bayes (One-Hot Vector)",
            svm_tf_idf,
            "SVM (Count Vector)"
        ]
    )

    # Disable Predict button if text input is empty
    predict_disabled = not user_input.strip()

    # A "Predict" button (enabled only if text input is provided)
    if st.button("Predict", disabled=predict_disabled):
        # For now, just display a placeholder until the model endpoints are ready.

        processed_data = [processing.test_precessing(user_input)]

        if(model_choice == svm_tf_idf):
            vectorizer = pickle.load_model_from_specified_path(model_path, "svm_tf_vectorizer")
            processed_data = vectorizer.transform(processed_data)

        prediction_text = predect.predict_target(pickle.load_model_from_specified_path(model_path, text_model_file_name[model_choice]), 
                                                 processed_data)

        # Show the result below the button
        st.subheader("Model Prediction Result")
        st.write(f"**Model Prediction**: <span style='color: green'>{prediction_text[0]}</span>", unsafe_allow_html=True)


elif page == "Image Classification":
    st.title("Image Classification - Multiple Models")

    st.subheader("Upload Image & Predict")
    uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)
        st.success("File uploaded successfully!")

    # Create a dropdown for 5 models
    model_choice = st.selectbox(
        "Select a Model:",
        [
            cnn,
            dnn,
            "KNN (K-Nearest Neighbors)",
            "SVM (Support Vector Machine)"
        ]
    )

    # Disable Predict button if no file is uploaded
    predict_disabled = uploaded_file is None

    # A "Predict" button (enabled only if a file is uploaded)
    if st.button("Predict", disabled=predict_disabled):
        # For now, just display a placeholder until the model endpoints are ready.
        label = {0 : 'food',1 : 'inside',2 : 'outside',3 : 'drink',4 : 'menu'}
        processed_data = processing.preprocess_image(uploaded_file)
        array = []

        if(model_choice == cnn or model_choice == dnn):
            array = np.expand_dims(processed_data, axis=0)
        else:
            array = processed_data.reshape(1, -1)
            print(array.shape)
            
        prediction_text = predect.predict_target(pickle.load_model_from_specified_path(model_path, img_model_file_name[model_choice]), array)

        # Show the result below the button
        st.subheader("Model Prediction Result")

        if(model_choice == cnn or model_choice == dnn):
            predicted_class = np.argmax(prediction_text, axis=1)
            st.write(f"**Model Prediction**: <span style='color: green'>{label[predicted_class[0]]}</span>", unsafe_allow_html=True)
        else:
            st.write(f"**Model Prediction**: <span style='color: green'>{label[prediction_text[0]]}</span>", unsafe_allow_html=True)
