import unittest
import joblib  # To load saved models
import numpy as np

class TestModels(unittest.TestCase):
    def setUp(self):
        """Load the trained models before running tests"""
        self.nb_model = joblib.load("models/naive_bayes.pkl")
        self.svm_model = joblib.load("models/svm.pkl")

    def test_nb_model_prediction(self):
        """Test if Naive Bayes model makes predictions"""
        sample_input = np.array([[1.5, 2.3, 3.1]])  # Example input
        prediction = self.nb_model.predict(sample_input)
        self.assertIsInstance(prediction[0], (int, np.integer))  # Should return an integer class label

    def test_svm_model_prediction(self):
        """Test if SVM model makes predictions"""
        sample_input = np.array([[0.5, 1.2, 2.8]])
        prediction = self.svm_model.predict(sample_input)
        self.assertIsInstance(prediction[0], (int, np.integer))

if __name__ == '__main__':
    unittest.main()
