import cv2
import numpy as np
import tensorflow as tf
import itertools
import os
# --- 1. SETUP MODEL ---
MODEL_PATH = "facenet_fullint.tflite"
# TODO: Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
# --- 2. SETUP DETECTOR ---
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade_path)
# --- 3. HELPER FUNCTIONS ---
# (Ensure you have defined 'detect_and_crop' and 'get_embedding' from previous steps)
# --- 4. MAIN LOGIC ---
image_files = ["Self_1.jpg", "Self_2.jpg", "Stranger_1.jpg", "Stranger_2.jpg"]
embeddings = {}
print("--- Processing Images ---")
# TODO: Loop through 'image_files':
# 1. Read image using cv2
image = cv2.imread(image_files)
# 2. Run 'detect_and_crop'
cropped_face, bounding_box = detect_and_crop(detector, image)
# 3. If face found, run 'get_embedding'
# 4. Store embedding in the 'embeddings' dictionary
if cropped_face is not None:
 embedding = get_embedding(interpreter, cropped_face)
 name = os.path.splitext(image_files)[0] # Remove .jpg from name
 embeddings[name] = embedding
 print(f"Processed: {name}")
else:
 print(f"No face found in {image_files}")
# --- 5. COMPARE PAIRS ---
results = []
print("\n--- Calculating Metrics ---")
# TODO: Use itertools.combinations to generate pairs
for name1, name2 in itertools.combinations(embeddings.keys(), 2):
 vec1 = embeddings[name1]
 vec2 = embeddings[name2]
 # TODO: Calculate Euclidean Distance (np.linalg.norm)
 euclidean_distance = np.linalg.norm(vec1 - vec2)

 # TODO: Calculate Cosine Similarity (dot_product / (norm_a * norm_b))
 dot_product = np.dot(vec1, vec2)
 norm_a = np.linalg.norm(vec1)
 norm_b = np.linalg.norm(vec2)
 cosine_similarity = dot_product / (norm_a * norm_b)

 # TODO: Append the pair name and both scores to the 'results' list
 results.append({"pair": f"{name1} vs {name2}", "euc": ..., "cos": ...})
 print(f"{name1} vs {name2} -> Euclidean: {euclidean_distance:.4f}, Cosine: {cos:.4f}")
# --- 6. OUTPUT TOP 2 MATCHES ---
# TODO: Sort 'results' by Euclidean Distance (Ascending: Lower is better)
# Print the Top 2
sorted_by_euclidean = sorted(results, key=lambda x: x["euc"])
print("\n--- Top 2 Matches by Euclidean Distance (Lower is Better) ---")
for r in sorted_by_euclidean[:2]:
 print(f"{r['pair']} -> {r['euc']:.4f}")
# TODO: Sort 'results' by Cosine Similarity (Descending: Higher is better)
# Print the Top 2
sorted_by_cosine_similarity = sorted(results, key=lambda x: x["cos"], reverse=True)
print("\n--- Top 2 Matches by Cosine Similarity (Higher is Better) ---")
for r in sorted_by_cosine_similarity[:2]:
 print(f"{r['pair']} -> {r['cos']:.4f}")
