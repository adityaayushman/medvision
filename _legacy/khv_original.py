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