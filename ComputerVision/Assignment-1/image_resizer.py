from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Open the image file
image = Image.open("C:\\Users\\kiran\\Downloads\\Yelp Photos\\images\\photos\\__gAh0oU6R4XRgVVHI9E7Q.jpg")

original_image_array = np.array(image)

print("original image")
print(original_image_array)

# Check the original size (600x337)
print(f"Original size: {image.size}")

# Resize the image to 224x224 pixels
resized_image = image.resize((224, 224))

# Save the resized image (optional)
resized_image.save("C:\\Users\\kiran\\Downloads\\Yelp Photos\\images\\resized_image_224x224.jpg")

# Check the new size (224x224)
print(f"New size: {resized_image.size}")

print("resized image")
print(resized_image)

image_array = img_to_array(resized_image)

print("resized_image")

print(image_array)

print(f"original_image shape: {original_image_array.shape}")
print(f"resized_image shape: {image_array.shape}")
