import streamlit as st

st.title("Text Classification - Multiple Models")

# 1. Create a dropdown for 5 models
model_choice = st.selectbox(
    "Select a Model:",
    [
        "SVM (TF-IDF Vector)",
        "SVM (Count Vector)",
        "Naive Bayes (Count Vector)",
        "Naive Bayes (TF-IDF Vector)",
        "Naive Bayes (One-Hot Vector)"
    ]
)

# Prepare two columns: one for input/button, one for the result
col_input, col_result = st.columns(2)

with col_input:
    st.subheader("Enter Text & Predict")
    # 2. A text area (multi-line) for user input
    user_input = st.text_area("Your text here:", height=100)
    
    # 3. A "Predict" button (placeholder for future model integration)
    if st.button("Predict"):
        # For now, just display a placeholder until the model endpoints are ready.
        prediction_text = "Prediction will be displayed here once the models are connected."

        # 4. Show the result in the second column
        with col_result:
            st.subheader("Model Prediction Result")
            st.write(f"**Selected Model**: {model_choice}")
            st.write(f"**User Input**: {user_input}")
            st.write(f"**Prediction**: {prediction_text}")
