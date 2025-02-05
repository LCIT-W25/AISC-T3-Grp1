import json
import pickle

# Step 1: Load the configuration file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Step 2: Select the correct model paths based on the config file
nb_model_path = "models/naive_bayes_tuned.pkl" if config["use_tuned_nb"] else "models/naive_bayes_untuned.pkl"
svm_model_path = "models/svm_tuned.pkl" if config["use_tuned_svm"] else "models/svm_untuned.pkl"

# Step 3: Load the selected models
with open(nb_model_path, "rb") as nb_file:
    nb_model = pickle.load(nb_file)

with open(svm_model_path, "rb") as svm_file:
    svm_model = pickle.load(svm_file)

# Step 4: Print which models were loaded
print(f"Loaded Naive Bayes Model: {'Tuned' if config['use_tuned_nb'] else 'Untuned'}")
print(f"Loaded SVM Model: {'Tuned' if config['use_tuned_svm'] else 'Untuned'}")