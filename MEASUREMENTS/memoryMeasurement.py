# MEMORY = 14.07 MB

import tracemalloc  # For tracking memory usage
import onnxruntime as ort
import numpy as np

# Start tracking memory
tracemalloc.start()

# Load the ONNX model
onnx_model_path = "model.onnx"
session = ort.InferenceSession(onnx_model_path)

# Prepare a dummy input for inference (adjust size to match your model's input)
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

# Get input and output names from the ONNX model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference with the dummy input
outputs = session.run([output_name], {input_name: dummy_input})

# Get the current, peak memory usage in MB
current_memory, peak_memory = tracemalloc.get_traced_memory()
current_memory = current_memory / (1024 * 1024)  # Convert to MB
peak_memory = peak_memory / (1024 * 1024)  # Convert to MB

# Stop tracking memory
tracemalloc.stop()

# Print the results
print(f"Current Memory Usage: {current_memory:.2f} MB")
print(f"Peak Memory Usage: {peak_memory:.2f} MB")
