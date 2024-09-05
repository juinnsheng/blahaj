from ultralytics import YOLO

# Paths to model configuration and dataset
model_path = "/Users/juinnshengna/Desktop/blahajdetection/yolov8n.pt"
data_path = "/Users/juinnshengna/Desktop/blahajdetection/Blahaj_det.v3i.yolov8/data.yaml"

# Define the function to retrain the model
def train_model():
    # Initialize the model
    model = YOLO(model_path)

    # Train the model
    model.train(
        data=data_path,
        epochs=15,
        imgsz=640
    )

    # Save the model after training
    model.save('yolov8_trained.pt')
    print("Model successfully trained and saved to 'yolov8_trained.pt'")

# Call the function to start training
train_model()
