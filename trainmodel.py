from ultralytics import YOLO

# Load the YOLOv8 model (this will automatically download the weights if not available locally)
model = YOLO('yolov8l.pt')  # No need to manually place weights in a directory

# Train the model on the custom dataset (plastic waste)
model.train(data='dataset/data.yaml', epochs=100, imgsz=640, batch=16)

# Optionally export the trained model
model.export(format='onnx')  # Example: exporting to ONNX format