import cv2
import numpy as np
def detect_and_crop(detector, image):
 """
 1. Detect face (Haar Cascade)
 2. Add 20% margin
 3. Crop and return (160, 160, 3) image
 """
 h, w, _ = image.shape

 # Haar Cascade works on Grayscale
 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 # Run Detection
 faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

 if len(faces) == 0:
 # print("No face detected!")
 return None, None
 # Get the largest face (Sorting by area: w * h)
 faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
 x, y, bw, bh = faces[0]

 # TODO: Calculate a 20% margin around the face
 # (Expand the box by 10% on each side)
 margin_x = int(bw * 0.1)
 margin_y = int(bh * 0.1)
 x_expanded = x - margin_x
 y_expanded = y - margin_y
w_expanded = bw + 2 * margin_x
 h_expanded = bh + 2 * margin_y

 # TODO: Ensure the expanded box is square (optional but recommended for FaceNet)
 # Hint: Find the largest dimension (width or height) and use that for both.
 square_size = max(w_expanded, h_expanded)
 # TODO: Center the new square box on the original face center
 face_center_x = x_expanded + w_expanded // 2
 face_center_y = y_expanded + h_expanded // 2
 x_square = face_center_x - square_size // 2
 y_square = face_center_y - square_size // 2
 # TODO: Handle boundary checks (ensure x, y < 0 or > image size are handled)
 x_final = max(0, x_square)
 y_final = max(0, y_square)
 w_final = min(square_size, w - x_final)
 h_final = min(square_size, h - y_final)
 # TODO: Crop the image
 cropped_image = image[y_final:y_final + h_final, x_final:x_final + w_final]
 # TODO: Resize the crop to (160, 160)
 cropped_image = cv2.resize(cropped_image, (160, 160))
 return cropped_image, (x_final, y_final, w_final, h_final)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Helper function to visualize
def show_bounding_box(image, bounding_box):
 x, y, w, h = bounding_box
 fig, ax = plt.subplots(1,1)

 # TODO: Convert BGR to RGB so colors look correct in Matplotlib
 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

 ax.imshow(image)
 ax.add_patch(Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))
 plt.show()
