# Hyperparameter Tuning for Different Models

# ---------------------------------------------
# K-Nearest Neighbors (KNN) Hyperparameter Tuning
# ---------------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import time

# Define the parameter grid for KNN
param_grid = {
    'n_neighbors': [3, 7, 11],  # Different values for k
    'metric': ['euclidean', 'manhattan']  # Different distance metrics
}

# Create the GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

start_time = time.time()
# Fit the model to the training data
grid_search.fit(x_train, y_train)
end_time = time.time()

# Calculate training time
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Display the best parameters from GridSearch
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Get the best model after GridSearch
best_knn = grid_search.best_estimator_

model_result(best_knn, x_train, y_train, x_test, y_test)
save_model(best_knn, "D:\\Models", "grid_search_knn_photo")
print("Model Saved Successfully")

# ---------------------------------------------
# Support Vector Machine (SVM) Hyperparameter Tuning
# ---------------------------------------------
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

# Define the parameter grid for SVM
param_grid = {
    'C': [0.1, 10],  # Test values for C
    'gamma': [0.01, 0.1],  # Test values for gamma
    'kernel': ['rbf']  # Kernel type
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=SVC(probability=True), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

start_time = time.time()
# Fit the model to the training data
grid_search.fit(x_train, y_train)
end_time = time.time()

# Calculate training time
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Display the best parameters from GridSearch
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Get the best model after GridSearch
best_svm = grid_search.best_estimator_

model_result(best_svm, x_train, y_train, x_test, y_test)
save_model(best_svm, "D:\\Models", "grid_search_svm")
print("Model Saved Successfully")

# ---------------------------------------------
# Convolutional Neural Network (CNN) Hyperparameter Tuning
# ---------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import time

# Function to create the CNN model
def build_cnn(activation='relu'):
    model = Sequential([
        Conv2D(32, (3, 3), activation=activation, input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation=activation),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation=activation),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation=activation),
        Dense(5, activation='softmax')  # Assuming 5 classes
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define hyperparameter grid manually
batch_sizes = [16, 32]
epochs_list = [5, 10]
activations = ['relu', 'tanh']

best_model = None
best_accuracy = 0
best_params = {}

start_time = time.time()
# Iterate through hyperparameter combinations
for batch_size in batch_sizes:
    for epochs in epochs_list:
        for activation in activations:
            print(f"\nTraining with batch_size={batch_size}, epochs={epochs}, activation={activation}")
            
            model = build_cnn(activation)
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), verbose=1)

            # Evaluate model on validation data
            val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)
            print(f'Validation Accuracy: {val_accuracy:.4f}')

            # Track the best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model
                best_params = {'batch_size': batch_size, 'epochs': epochs, 'activation': activation}
end_time = time.time()

# Print best parameters
print("\nBest Model Hyperparameters:")
print(best_params)
print(f"Best Validation Accuracy: {best_accuracy:.4f}")

# Save the best model
best_model.save("best_cnn_model.h5")
print("CNN Model Saved Successfully")
