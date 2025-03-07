import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

class Interpertable:

    def __init__(self):
        pass # will create the class instance

    def explain_with_lime_for_image(self, model, img_array, top_labels=1, num_samples=1000):
        """Generates LIME explanation for image classification."""
        
        explainer = lime_image.LimeImageExplainer()
        
        def model_predict(input_img):
            """Helper function to get model predictions."""
            preds = model.predict(input_img)
            return preds
        
        explanation = explainer.explain_instance(
            img_array[0].astype('double'), 
            model_predict, 
            top_labels=top_labels, 
            hide_color=0, 
            num_samples=num_samples
        )

        # Get explanation mask
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest= False
        )

        # Convert superpixel mask into a color image
        highlighted_img = mark_boundaries(temp, mask)

        return highlighted_img
    

    # LIME Explanation for RNN
    def lime_explanation_rnn(self, model, sample_text_sequence, reverse_label_map, max_len):
        """
        Generates a LIME explanation for a single RNN input.
        """
        class_names = [reverse_label_map[i] for i in range(len(reverse_label_map))]
        explainer = LimeTextExplainer(class_names=class_names)

        def predict_fn(text_instances):
            sequences = [np.array(list(map(int, text.split()))) for text in text_instances]
            sequences_padded = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant') for seq in sequences])
            return model.predict(sequences_padded)

        exp = explainer.explain_instance(
            " ".join(map(str, sample_text_sequence)),
            predict_fn,
            num_features=10,
            top_labels=5
        )
        return exp

    # LIME Explanation for GNN
    def explain_single_instance_with_lime(self, model, x_test_instance, num_features=10):
        """
        Explains a single instance from x_test using LIME.
        """
        x_test_instance_flattened = x_test_instance.reshape(1, -1)

        def predict_fn(x):
            x_reshaped = x.reshape(x.shape[0], x_test_instance.shape[0], x_test_instance.shape[1])
            return model.predict(x_reshaped)

        explainer = LimeTabularExplainer(
            training_data=x_train.reshape(x_train.shape[0], -1),
            mode="classification",
            class_names=list(reverse_label_map.values()),
            feature_names=[f"Feature {i}" for i in range(x_test_instance_flattened.shape[1])],
            discretize_continuous=True
        )

        explanation = explainer.explain_instance(
            x_test_instance_flattened[0],
            predict_fn,
            num_features=num_features,
            top_labels=5
        )

        return explanation

