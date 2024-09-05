from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO('yolov8_trained.pt')

# Start the webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform prediction with YOLOv8
    # Lower the confidence threshold for better detection
    results = model.predict(source=frame, conf=0.3, save=False, show=False)  # Use a lower confidence threshold

    # Check if results are available
    if results:
        annotated_frame = results[0].plot()  # Render the predictions onto the frame

        # Display the frame with detected objects
        cv2.imshow("Detected Objects", annotated_frame)
    else:
        print("No results found")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the results (optional)
print(results)
