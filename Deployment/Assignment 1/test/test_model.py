import unittest
import joblib
import numpy as np
from tensorflow.keras.models import load_model

class TestModels(unittest.TestCase):
    def setUp(self):
        """Load the trained models before running tests"""
        self.nb_model = joblib.load("models/naive_bayes.pkl")  # Load Naive Bayes model
        self.svm_model = joblib.load("models/svm.pkl")         # Load SVM model
        self.knn_model = joblib.load("models/knn.pkl")         # Load KNN model
        self.dnn_model = load_model("models/dnn.h5")          # Load DNN model

    def test_nb_model_prediction(self):
        """Test if Naive Bayes model makes predictions"""
        sample_input = np.array([[1.5, 2.3, 3.1]])  # Example input
        prediction = self.nb_model.predict(sample_input)
        self.assertIsInstance(prediction[0], (int, np.integer))  # Should return a class label

    def test_svm_model_prediction(self):
        """Test if SVM model makes predictions"""
        sample_input = np.array([[0.5, 1.2, 2.8]])  # Example input
        prediction = self.svm_model.predict(sample_input)
        self.assertIsInstance(prediction[0], (int, np.integer))  # Should return a class label

    def test_knn_model_prediction(self):
        """Test if KNN model makes predictions"""
        sample_input = np.array([[6, 7, 8]])  # Example input
        prediction = self.knn_model.predict(sample_input)
        self.assertIsInstance(prediction[0], (int, np.integer))  # Should return a class label

    def test_dnn_model_prediction(self):
        """Test if DNN model makes predictions"""
        sample_input = np.array([[6, 7, 8]])  # Example input
        prediction = self.dnn_model.predict(sample_input)
        self.assertIsInstance(prediction[0][0], (float, np.floating))  # Should return a probability score

if __name__ == "__main__":
    unittest.main()
