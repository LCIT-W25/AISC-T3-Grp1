import unittest
import joblib  # To load saved models
import numpy as np

class TestModels(unittest.TestCase):
    def setUp(self):
        """Load the trained models before running tests"""
        try:
            self.nb_model = joblib.load("models/naive_bayes.pkl")
            self.svm_model = joblib.load("models/svm.pkl")
        except FileNotFoundError:
            self.fail("Model file not found!")
        except Exception as e:
            self.fail(f"Error loading models: {str(e)}")

    def test_nb_model_prediction(self):
        """Test if Naïve Bayes model makes predictions"""
        sample_input = np.random.rand(1, self.nb_model.n_features_in_)  # Match feature size
        prediction = self.nb_model.predict(sample_input).item()  # Ensure scalar output
        self.assertIsInstance(prediction, (int, np.integer))  # Should return an integer class label

    def test_svm_model_prediction(self):
        """Test if SVM model makes predictions"""
        sample_input = np.random.rand(1, self.svm_model.n_features_in_)
        prediction = self.svm_model.predict(sample_input).item()
        self.assertIsInstance(prediction, (int, np.integer))

if __name__ == '__main__':
    unittest.main()
