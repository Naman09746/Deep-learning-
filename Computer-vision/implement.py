import cv2
import torch
from gesture_model import GestureModel  # Assuming you have defined the model
import numpy as np

# Load the trained model
model = GestureModel(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('best_gesture_model.pth'))
model.eval()

# OpenCV setup
cap = cv2.VideoCapture(0)  # Use webcam for live video capture

# Create a window for displaying results
cv2.namedWindow("Gesture Control")

while True:
    ret, frame = cap.read()  # Capture a frame
    if not ret:
        break
    
    # Preprocess the frame (resize and normalize it)
    img = cv2.resize(frame, (128, 128))  # Resize to match the model's input size
    img = img / 255.0  # Normalize
    img_tensor = torch.Tensor(img).unsqueeze(0).to(device)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
    
    # Map the predicted class to a gesture
    predicted_gesture = class_names[predicted_class.item()]
    print(f"Predicted Gesture: {predicted_gesture}")
    
    # Display the predicted gesture
    cv2.putText(frame, f"Gesture: {predicted_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Map gesture to smart home actions
    if predicted_gesture == "thumb":  # Turn on light
        print("Turning on light...")
        # Add code to trigger smart home light
    elif predicted_gesture == "fist":  # Turn off light
        print("Turning off light...")
        # Add code to trigger smart home light off
    elif predicted_gesture == "index":  # Adjust thermostat
        print("Adjusting thermostat...")
        # Add code to adjust thermostat
    elif predicted_gesture == "palm":  # Play/Pause music
        print("Playing music...")
        # Add code to control music
    elif predicted_gesture == "c":  # Open/close blinds
        print("Controlling blinds...")
        # Add code to control blinds

    # Show the frame with the predicted gesture
    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
