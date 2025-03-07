class Predect:

    def __init__(self):
        pass # will create the class instance

    def predict_target(self, model, preprocessed_data):
            
        # Make predictions
        prediction = model.predict(preprocessed_data)
            
        # return label[prediction]
        return prediction