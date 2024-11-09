import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read an image from file
image = cv2.imread('path_to_your_image.jpg', cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)  # Lower and upper thresholds for edge detection

# Display the original image and the edge-detected image using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection (Canny)")
plt.axis('off')

plt.show()
