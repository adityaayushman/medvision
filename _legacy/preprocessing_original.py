import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Image
image = cv2.imread(r"C:\Users\ADITYA\Documents\inrenship project\MED VISION\download.jpg")

# Resize
image = cv2.resize(image, (512, 512))

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur (Noise Removal)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Histogram Equalization
hist_eq = cv2.equalizeHist(blur)

# CLAHE Enhancement
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(hist_eq)

# Edge Detection
edges = cv2.Canny(enhanced, 50, 150)

# Display
titles = [
    "Original",
    "Gray",
    "Blur",
    "Enhanced",
    "Edges"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    gray,
    blur,
    enhanced,
    edges
]

plt.figure(figsize=(15,5))

for i in range(len(images)):
    plt.subplot(1,5,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.show()

kernel = np.ones((5,5), np.uint8)

# Dilation
dilation = cv2.dilate(edges, kernel, iterations=1)

# Erosion
erosion = cv2.erode(edges, kernel, iterations=1)

# Opening
opening = cv2.morphologyEx(
    edges,
    cv2.MORPH_OPEN,
    kernel
)

# Closing
closing = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel
)

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(dilation, cmap='gray')
plt.title("Dilation")

plt.subplot(2,2,2)
plt.imshow(erosion, cmap='gray')
plt.title("Erosion")

plt.subplot(2,2,3)
plt.imshow(opening, cmap='gray')
plt.title("Opening")

plt.subplot(2,2,4)
plt.imshow(closing, cmap='gray')
plt.title("Closing")

plt.show()

ret, segmented = cv2.threshold(
    enhanced,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(enhanced, cmap='gray')
plt.title("Enhanced Image")

plt.subplot(1,2,2)
plt.imshow(segmented, cmap='gray')
plt.title("Segmented Image")

plt.show()

contours, _ = cv2.findContours(
    segmented,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > 500:

        x,y,w,h = cv2.boundingRect(cnt)

        cv2.rectangle(
            output,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(output,
                        cv2.COLOR_BGR2RGB))
plt.title("Detected Regions")
plt.axis("off")
plt.show()