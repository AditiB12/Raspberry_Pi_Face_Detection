# Raspberry_Pi_Face_Detection
* Used the Picamera2 library to access the Raspberry Pi to access the camera hardware, allowing me to grab high-speed frames as NumPy arrays and pass them directly to OpenCV for detection.

## A summary of everything this project does in detail is below:
1. The real-time detection script initializes the camera, captures frames in a loop, and applies the Haar Cascade detector.
2. 2. A crop and detect function calculates a square bounding box to see the entire head (hair, ears, chin). This function returns a 160 x 160 image for inference. 
3. The 160 x 160 face image is passed to the TFLite model to get a 512-dimensional embedding vector. The 
vector is used to calculate the Euclidean distance and cosine similarity between the images of upto 2 different people.

#### Explanation of euclidean distance and Cosine similarity copied from the University of Illinois Urbana Champaign's ECE 479: IoT Systems curriculum: 
Euclidean Distance vs. Cosine Similarity: While Euclidean distance is common, Cosine Similarity is the standard for comparing embeddings because it measures the angle between vectors, ignoring lighting-induced magnitude changes.
Euclidean: Lower is better (0.0 = exact match).
Cosine: Higher is better (1.0 = exact match).
Final Task: Collect 4 images:
2 images of yourself (Self_A, Self_B)
2 images of a stranger (Stranger_A, Stranger_B)
Write a script that calculates the Euclidean Distance between all 6 possible pairs.
Expected: (Self_A, Self_B) should be the smallest distance.
Expected: (Self_A, Stranger_A) should be significantly larger.



