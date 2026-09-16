import tflite_runtime.interpreter as tflite
def get_embedding(interpreter, face_img):
 # Get input and output details
 input_details = interpreter.get_input_details()
 output_details = interpreter.get_output_details()

 # 1. Preprocess (Standardize to [-1, 1])
 # Ensure input is float32 for the math
 face = face_img.astype('float32')
 face = (face - 127.5) / 127.5

 # 2. Check Input Quantization
 # If the model expects int8, we must quantize the input manually
 if input_details[0]['dtype'] == np.int8:
 scale, zero_point = input_details[0]['quantization']
 # TODO: Apply quantization formula
 face = (face / scale) + zero_point

 # TODO: Clip to int8 range (-128, 127) and cast type
 face = np.clip(face, -128, 127).astype(np.int8)
# 3. Add Batch Dimension (1, 160, 160, 3)
 face = np.expand_dims(face, axis=0)

 # 4. Run Inference
 # TODO: Set input tensor
 interpreter.set_tensor(input_details[0]['index'], face)

 # TODO: Invoke interpreter
 interpreter.invoke()

 # TODO: Get output tensor
 output = interpreter.get_tensor(output_details[0]['index'])

 # 5. Check Output Dequantization
 if output_details[0]['dtype'] == np.int8:
 scale, zero_point = output_details[0]['quantization']
 # TODO: Convert int8 output back to float32
 output = (output.astype('float32') - zero_point) * scale

 return output.flatten()
