import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image using PIL and convert it to a NumPy array
image_path = r'C:\Users\kiran\Downloads\Yelp Photos\images\photos\__RMLkfCq_spMU1Q5TDHnA.jpg'  # Replace with your image file path
img = Image.open(image_path)
image_array = np.array(img.resize((224,224)))

print("Original Image array" ,image_array)

image_array = image_array/255.0

print("Normalized Image array" ,image_array)

# Print the shape of the original image
print("Original Image Shape:", image_array.shape)  # (height, width, channels)

# Ensure the image has 3 channels (RGB). Convert if needed.
# if image_array.ndim == 2:  # If grayscale, convert to RGB
#     image_array = np.stack((image_array,) * 3, axis=-1)

# Expand dimensions to create a batch of one image
image_batch = np.expand_dims(image_array, axis=0)
print("Image Shape After Adding Batch Dimension:", image_batch.shape)  # (1, height, width, channels)

# Define augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Random horizontal flip
    tf.keras.layers.RandomRotation(0.2),      # Rotate by 20% of 360°
    tf.keras.layers.RandomZoom(0.2),          # Zoom in/out
])

# Apply augmentation
augmented_images = [data_augmentation(image_batch)[0].numpy().astype(np.float32) for _ in range(10)]

# Display original and augmented images
plt.figure(figsize=(15, 5))
plt.subplot(1, 11, 1)
plt.imshow(image_array)
plt.title("Original")
plt.axis("off")

for i in range(10):
    plt.subplot(1, 11, i + 2)
    print(augmented_images[i].shape)
    print(augmented_images[i])
    plt.imshow(augmented_images[i])
    plt.title("Augmented")
    plt.axis("off")

plt.show()
