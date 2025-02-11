#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from lime import lime_image
from skimage.segmentation import mark_boundaries
from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Flatten
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam


# In[27]:


photos_csv = r'C:\Users\User\Documents\GIT Main\AISC-T3-Grp1\Computer Vision\Assignment 2\photos.csv'
photos_folder = r'C:\Users\User\Documents\GIT Main\AISC-T3-Grp1\Computer Vision\Assignment 2\photos'
# Load dataset
df = pd.read_csv(photos_csv)


# In[28]:


df


# In[29]:


def bar_plot(data):
    class_counts = pd.Series(data).value_counts()

    # Plot bar chart
    class_counts.plot(kind="bar", color=["green", "red", "blue", "orange", "yellow"], edgecolor="black", figsize=(8, 6))

    # Add values on top of each bar
    for i, v in enumerate(class_counts):
        plt.text(i, v + 0.2, str(v), ha="center", fontsize=12)
        
        
    print("class_count : ", class_counts)
    # Customize plot
    plt.xlabel("Sentiment Labels")
    plt.ylabel("Frequency")
    plt.title("Sentiment Distribution in Traget ")
    plt.xticks(rotation=0)  # Keep labels readable
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()


# In[30]:


bar_plot(df['label'])


# In[31]:


# Separate majority and minority classes
df_majority_food = df[df['label'] == 'food']
df_majority_inside = df[df['label'] == 'inside']
df_majority_outside = df[df['label'] == 'outside']
df_majority_drink = df[df['label'] == 'drink']
df_minority_menu = df[df['label'] == 'menu']

# Undersample majority classes to 600 samples each
df_majority_food_downsampled = resample(df_majority_food, replace=False, n_samples=300, random_state=42)
df_majority_inside_downsampled = resample(df_majority_inside, replace=False, n_samples=300, random_state=42)
df_majority_outside_downsampled = resample(df_majority_outside, replace=False, n_samples=300, random_state=42)
df_majority_drink_downsampled = resample(df_majority_drink, replace=False, n_samples=300, random_state=42)

# Oversample minority class to 600 samples
df_minority_menu_upsampled = resample(df_minority_menu, replace=True, n_samples=300, random_state=42)

# Combine all classes into a balanced DataFrame
df_balanced = pd.concat([df_majority_food_downsampled, df_majority_inside_downsampled, df_majority_outside_downsampled, df_majority_drink_downsampled, df_minority_menu_upsampled])

# Shuffle the DataFrame
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


# In[32]:


df_balanced


# In[33]:


bar_plot(df_balanced['label'])


# In[34]:


df = df_balanced


# In[35]:


# Build a dictionary to map photo_id to the image file paths
photo_id_to_path = {}
for file in os.listdir(photos_folder):
    # Assuming the photo_id is in the file name (e.g., photo_id.jpg)
    photo_id = file.split('.')[0] # Adjust based on your file naming convention
    photo_id_to_path[photo_id] = os.path.join(photos_folder, file)


# In[36]:


# Open the image
img = Image.open("C:\\Users\\User\\Documents\\GIT Main\\AISC-T3-Grp1\\Computer Vision\\Assignment 2\\photos\\zsvj7vloL4L5jhYyPIuVwg.jpg")

# Get the image size (width, height)
width, height = img.size
print(f"Width: {width}, Height: {height}")


# In[39]:


# Function to preprocess images (resize to 224x224 and normalize)
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        return img_array / 255.0  # Normalize the image
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Image path: {img_path}")
        return None

# Preprocess images for each photo_id
x_images = []
y_labels = []

for _, row in df.iterrows():
    photo_id = row['photo_id']
    label = row['label']
    
    # Get the image path based on the photo_id
    img_path = photo_id_to_path.get(photo_id)
    
    if img_path:
        # Preprocess the image
        img = preprocess_image(img_path)
        if(img is not None) :
            x_images.append(img)
            y_labels.append(label)

# Convert lists to numpy arrays for further processing
print(len(x_images))

x_images = np.array(x_images)

print(x_images)
print(y_labels)


print("\nShape of the image array: ", x_images.shape)
print("length of target: ", len(y_labels))


# In[40]:


label_mapping = {'food' : 0, 'inside' : 1, 'outside' : 2, 'drink' : 3, 'menu' : 4}
reverse_label_mapping = {0 : 'food',1 : 'inside',2 : 'outside',3 : 'drink',4 : 'menu'}


# In[41]:


# Map the labels using the custom dictionary
y = np.array([label_mapping[label] for label in y_labels])

# Flatten the images
x = x_images.reshape(x_images.shape[0], -1)

print("Shape of the image array: ",x.shape)
print("Shape of the target array: ",y.shape)

print(y[:100])


# In[42]:


from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# Apply EfficientNet preprocessing to the entire batch of images
preprocessed_images = effnet_preprocess(x_images)


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(preprocessed_images, y, test_size=0.2, random_state=42)
print("Shape of x_train: ", x_train.shape)  # Should print (num_samples, 224, 224, 3)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)


# In[44]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

def explain_with_lime(model, image_array, model_type="vgg", top_labels=5, num_samples=1000):
    """
    Generates LIME explanations for a given image (numpy array).
    
    Parameters:
    - model: Trained TensorFlow model (VGG16 or EfficientNet)
    - image_array: Single image as a numpy array (shape: (224, 224, 3))
    - model_type: "vgg" for VGG16, "efficientnet" for EfficientNet
    - top_labels: Number of labels to explain
    - num_samples: Number of perturbations for LIME
    
    Returns:
    - Displays LIME explanation heatmap
    """

    # Expand dims to match the model input shape (1, 224, 224, 3)
    img_array = np.expand_dims(image_array, axis=0)

    # Apply the correct preprocessing
    if model_type == "vgg":
        img_array = vgg_preprocess(img_array)  # Preprocess for VGG16
    elif model_type == "efficientnet":
        img_array = effnet_preprocess(img_array)  # Preprocess for EfficientNet

    # Define prediction function for LIME
    def model_predict(images):
        images = np.array([
            effnet_preprocess(img) if model_type == "efficientnet" else vgg_preprocess(img)
            for img in images
        ])
        return model.predict(images)

    # Initialize LIME Explainer
    explainer = lime_image.LimeImageExplainer()

    # Explain the prediction
    explanation = explainer.explain_instance(
        image_array.astype('double'),  # Convert to double for LIME
        model_predict,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples
    )

    # Get the top predicted label
    top_label = explanation.top_labels[0]

    # Get the explanation mask (increase num_features for better visibility)
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=20, hide_rest=False  # Increased num_features for better visibility
    )

    # Display the result
    plt.figure(figsize=(8, 8))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f"LIME Explanation for {model_type.upper()} - Class {top_label}")
    plt.axis("off")
    plt.show()


# In[45]:


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


# In[46]:


# Load EfficientNetB0 without top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Keep it frozen for now


# In[47]:


image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)


# In[48]:


model = Model(inputs=image_input, outputs=final_output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[49]:


# Fine-tune the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


# In[50]:


# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)


# In[51]:


# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)


# In[52]:


# Reverse the label mapping for confusion matrix
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    # Step 1: Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Step 2: Convert the confusion matrix to a DataFrame for better visualization
    cm_df = pd.DataFrame(cm, index=[reverse_label_mapping[i] for i in range(len(cm))], 
                         columns=[reverse_label_mapping[i] for i in range(len(cm))])
    
    # Print confusion matrix dataframe
    print("Confusion Matrix:\n", cm_df)

    # Step 3: Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues", xticklabels=cm_df.columns, yticklabels=cm_df.columns, cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


# In[53]:


plot_confusion_matrix(y_train, y_train_pred_labels )


# In[54]:


plot_confusion_matrix(y_test, y_test_pred_labels)


# In[55]:


def predict_image(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Error loading image"
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return label_mapping[predicted_class]


# In[56]:


def plot_roc_curve(y_true, y_pred_probs):
    # Initialize LabelBinarizer for multi-class handling
    lb = LabelBinarizer()
    
    # Binarize the true labels (convert to one-hot encoding)
    y_true_bin = lb.fit_transform(y_true)
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Iterate through each class (works for multi-class)
    for i in range(y_true_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves in one plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC for each class
    for i in range(y_true_bin.shape[1]):
        label = lb.classes_[i]
        plt.plot(fpr[i], tpr[i], label=f'{reverse_label_mapping[label]} (AUC = {roc_auc[i]:.2f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')

    # Labeling and Title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print AUC values for each class
    for i in range(y_true_bin.shape[1]):
        print(f'AUC for class {reverse_label_mapping[lb.classes_[i]]}: {roc_auc[i]:.2f}')


# In[57]:


plot_roc_curve(y_train, y_train_pred ) 


# In[58]:


plot_roc_curve(y_test, y_test_pred) 


# In[59]:


# Calculate and print the classification report for training data
print("Classification Report for Training Data:")
print(classification_report(y_train, y_train_pred_labels, target_names=[reverse_label_mapping[i] for i in range(5)]))


# In[60]:


# Calculate and print the classification report for test data
print("Classification Report for Test Data:")
print(classification_report(y_test, y_test_pred_labels, target_names=[reverse_label_mapping[i] for i in range(5)]))


# lime

# In[61]:


explain_with_lime(model, x_test[0], model_type="efficientnet")


# ### Unfreezing the last layers

# In[62]:


tf.keras.backend.clear_session()


# In[63]:


# One-hot encode the target labels for training and testing data
y_train_onehot = to_categorical(y_train, num_classes=5)
y_test_onehot = to_categorical(y_test, num_classes=5)


# In[64]:


for layer in model.layers[-10:]:  # Unfreeze the last 10 layers (you can adjust this number)
    layer.trainable = True


# In[65]:


model.compile(optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# In[66]:


# Retrain the model with the unfreezed layers
history_finetuned = model.fit(x_train, y_train_onehot, 
                              epochs=10,  # Adjust the number of epochs for fine-tuning
                              batch_size=32,
                              validation_data=(x_test, y_test_onehot))


# In[67]:


y_train_pred_probs = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred_probs, axis=1)


# In[68]:


y_test_pred_probs = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred_probs, axis=1)


# In[69]:


# Classification Report for Training Data
print("\nClassification Report for Training Data:")
print(classification_report(np.argmax(y_train_onehot, axis=1), y_train_pred_labels))


# In[70]:


# Classification Report for Test Data
print("\nClassification Report for Test Data:")
print(classification_report(np.argmax(y_test_onehot, axis=1), y_test_pred_labels))


# In[71]:


# Plot Confusion Matrix for Training Data
print("\nConfusion Matrix for Training Data:")
plot_confusion_matrix(np.argmax(y_train_onehot, axis=1), y_train_pred_labels)


# In[72]:


# Plot Confusion Matrix for Test Data
print("\nConfusion Matrix for Test Data:")
plot_confusion_matrix(np.argmax(y_test_onehot, axis=1), y_test_pred_labels)


# In[73]:


# Plot ROC Curve for Test Data
print("\nROC Curve for Test Data:")
plot_roc_curve(np.argmax(y_test_onehot, axis=1), y_test_pred_probs)


# In[74]:


# Plot ROC Curve for Training Data (Optional)
print("\nROC Curve for Training Data:")
plot_roc_curve(np.argmax(y_train_onehot, axis=1), y_train_pred_probs)


# lime

# In[75]:


explain_with_lime(model, x_test[0], model_type="efficientnet")


# efficientnet with all freeze layers and Image augumentaion 

# In[76]:


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load EfficientNetB0 without top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Keep it frozen for now

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator with augmentation for the training data
train_datagen = ImageDataGenerator(
    rotation_range=40,  # Random rotation between 0 and 40 degrees
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Filling mode for newly created pixels
)

# No augmentation for the test data, just use it as it is
test_datagen = ImageDataGenerator()

# Apply the augmentation to your training and testing data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

# Fine-tune the model with augmented images
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

print("confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels )
print("confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

print("ROC for train data")
plot_roc_curve(y_train, y_train_pred ) 
print("ROC for test data")
plot_roc_curve(y_test, y_test_pred ) 


# lime

# In[77]:


explain_with_lime(model, x_test[0], model_type="efficientnet")


# efficientnet with unfreeze all layers and image augumentanion and gaussian blur

# lime

# In[79]:


# explain_with_lime(model, x_test[0], model_type="efficientnet")


# efficinetnet with unfreeze some layers

# In[80]:


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load EfficientNetB0 without top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Unfreeze the base model

# Unfreeze some layers of the base model
# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[:-20]:  # Unfreeze the last 20 layers, keep the earlier ones frozen
    layer.trainable = False

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=True)  # Training=True to fine-tune
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator with augmentation for the training data
train_datagen = ImageDataGenerator(
    rotation_range=40,  # Random rotation between 0 and 40 degrees
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Filling mode for newly created pixels
)

# No augmentation for the test data, just use it as it is
test_datagen = ImageDataGenerator()

# Apply the augmentation to your training and testing data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

# Fine-tune the model with augmented images
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Confusion Matrix
print("Confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels)

print("Confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

# ROC Curve
print("ROC for train data")
plot_roc_curve(y_train, y_train_pred)

print("ROC for test data")
plot_roc_curve(y_test, y_test_pred)


# lime

# In[81]:


explain_with_lime(model, x_test[0], model_type="efficientnet")


# efficientnet with unfreeze all layers and image augumentation

# In[83]:


import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load EfficientNetB0 without top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze all layers for fine-tuning
base_model.trainable = True

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=True)  # Set training=True for fine-tuning
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator with augmentation for the training data
train_datagen = ImageDataGenerator(
    rotation_range=40,  # Random rotation between 0 and 40 degrees
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Filling mode for newly created pixels
)

# No augmentation for the test data, just use it as it is
test_datagen = ImageDataGenerator()

# Apply the augmentation to your training and testing data
train_generator = train_datagen.flow(x_train, y_train, batch_size=16)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=16)

# Fine-tune the model with augmented images
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

print("Confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels)

print("Confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

print("ROC for train data")
plot_roc_curve(y_train, y_train_pred)

print("ROC for test data")
plot_roc_curve(y_test, y_test_pred)


# lime

# In[84]:


explain_with_lime(model, x_test[0], model_type="efficientnet")


# In[85]:


from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

# Apply VGG16 preprocessing to the entire batch of images
preprocessed_images = vgg_preprocess(x_images)  # This scales and subtracts the mean values


# In[86]:


x_train, x_test, y_train, y_test = train_test_split(preprocessed_images, y, test_size=0.2, random_state=42)
print("Shape of x_train: ", x_train.shape)  # Should print (num_samples, 224, 224, 3)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)


# vgg freeze all layers

# In[87]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers of VGG16 (no fine-tuning)
base_model.trainable = False  

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=False)  # Use training=False to keep frozen layers stable
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator (without augmentation for training data)
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# Apply the generator to your training and testing data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

# Train the model (only the custom dense layers will be updated)
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Confusion Matrix
print("Confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels)

print("Confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

# ROC Curve
print("ROC for train data")
plot_roc_curve(y_train, y_train_pred)

print("ROC for test data")
plot_roc_curve(y_test, y_test_pred)


# lime

# In[88]:


explain_with_lime(model, x_test[0], model_type="vgg")


# vgg freeze all layears with image augumentation

# In[89]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers of VGG16
base_model.trainable = False  

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=False)  # Keep batch norm layers in inference mode
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator with augmentation for the training data
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for the test data
test_datagen = ImageDataGenerator()

# Apply augmentation to training data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

# Train the model (only the custom dense layers will be updated)
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Confusion Matrix
print("Confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels)

print("Confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

# ROC Curve
print("ROC for train data")
plot_roc_curve(y_train, y_train_pred)

print("ROC for test data")
plot_roc_curve(y_test, y_test_pred)


# lime

# In[90]:


explain_with_lime(model, x_test[0], model_type="vgg")


# vgg with some unfreeze layers and image augumentation

# In[91]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last few layers of the base model and freeze the rest
for layer in base_model.layers[:-4]:  # Freeze all layers except the last 4
    layer.trainable = False

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=True)  # Training=True to fine-tune the unfrozen layers
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator with augmentation for the training data
train_datagen = ImageDataGenerator(
    rotation_range=40,  # Random rotation between 0 and 40 degrees
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Filling mode for newly created pixels
)

# No augmentation for the test data, just use it as it is
test_datagen = ImageDataGenerator()

# Apply the augmentation to your training and testing data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

# Fine-tune the model with augmented images
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Confusion Matrix
print("Confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels)

print("Confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

# ROC Curve
print("ROC for train data")
plot_roc_curve(y_train, y_train_pred)

print("ROC for test data")
plot_roc_curve(y_test, y_test_pred)


# lime

# In[92]:


explain_with_lime(model, x_test[0], model_type="vgg")


# unfreeze all layers

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load VGG16 without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # No freezing, fine-tune the entire model

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=True)  # Training=True to fine-tune
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
final_output = Dense(5, activation='softmax')(x)

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator (without augmentation for the training data)
train_datagen = ImageDataGenerator()

# No augmentation for the test data, just use it as it is
test_datagen = ImageDataGenerator()

# Apply the generator to your training and testing data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

# Fine-tune the model with the original images (no augmentation)
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Confusion Matrix
print("Confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels)

print("Confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

# ROC Curve
print("ROC for train data")
plot_roc_curve(y_train, y_train_pred)

print("ROC for test data")
plot_roc_curve(y_test, y_test_pred)


# lime

# In[ ]:


explain_with_lime(model, x_test[0], model_type="vgg")


# unfreeze all layers and image augumentation

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load VGG16 without top layers (include_top=False) for fine-tuning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# No freezing of any layers (all layers are trainable)
base_model.trainable = True

# Build the custom model
image_input = Input(shape=(224, 224, 3))
x = base_model(image_input, training=True)  # Training=True to ensure all layers are trainable
x = GlobalAveragePooling2D()(x)  # Pooling the feature map output from the base model
x = Dense(256, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
x = Dense(128, activation='relu')(x)  # Another fully connected layer
x = Dropout(0.3)(x)  # Dropout
final_output = Dense(5, activation='softmax')(x)  # Output layer with softmax for multi-class classification

model = Model(inputs=image_input, outputs=final_output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the ImageDataGenerator with augmentation for the training data
train_datagen = ImageDataGenerator(
    rotation_range=40,  # Random rotation between 0 and 40 degrees
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shearing transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Filling mode for newly created pixels
)

# No augmentation for the test data, just use it as it is
test_datagen = ImageDataGenerator()

# Apply the augmentation to your training and testing data
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=32)

# Fine-tune the model with augmented images
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Predict the labels for the training set
y_train_pred = model.predict(x_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)

# Predict the labels for the test set
y_test_pred = model.predict(x_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Confusion Matrix
print("Confusion Matrix for train data")
plot_confusion_matrix(y_train, y_train_pred_labels)

print("Confusion Matrix for test data")
plot_confusion_matrix(y_test, y_test_pred_labels)

# ROC Curve
print("ROC for train data")
plot_roc_curve(y_train, y_train_pred)

print("ROC for test data")
plot_roc_curve(y_test, y_test_pred)

