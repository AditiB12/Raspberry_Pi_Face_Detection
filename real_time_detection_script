import cv2
import time
from picamera2 import Picamera2
# --- CONFIGURATION ---
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade_path)
# 1. Verify Detector
if detector.empty():
 print("Error: Could not load Haar Cascade XML.")
 exit()
else:
 print("Face Detector Loaded")
# 2. Initialize Camera Module 3 via Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)})
picam2.configure(config)
picam2.start()
time.sleep(2) # Warm up camera
print("Camera initialized. Press 'q' to quit.")
# 3. Main Loop
try:
 while True:
 frame = picam2.capture_array()
 # Convert to grayscale for detection (Required by Haar Cascade)
 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 # Detect Faces
 # scaleFactor=1.1: Reduces image size by 10% each pass
 # minNeighbors=5: Higher = less false positives
 faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
 # Draw rectangles around faces
 for (x, y, w, h) in faces:
 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 # Convert RGB to BGR for OpenCV display
 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
 cv2.imshow('Simple Face Test', frame_bgr)
 # Press 'q' to exit
 if cv2.waitKey(1) & 0xFF == ord('q'):
 break
except KeyboardInterrupt:
 pass
# Cleanup
picam2.stop()
cv2.destroyAllWindows()
# Show bounding box of face detected by MTCNN in the captured image
from PIL import Image
import matplotlib.pyplot as plt
image = Image.open("Simple_Face_Test.png")
plt.imshow(image)

