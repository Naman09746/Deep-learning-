import torch
import cv2
import pyautogui
from gesture_model import GestureModel
import numpy as np

# Load the trained model
model = GestureModel(num_classes=11)  # Change the number of classes accordingly
model.load_state_dict(torch.load('best_gesture_model.pth'))
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use your webcam (0 for the default camera)

# Define screen dimensions for mapping gestures to screen space
screen_width, screen_height = pyautogui.size()  # Get the screen width and height

# Mapping gesture IDs to mouse actions
gesture_map = {
    1: "move_up",    # Example: Palm Gesture (Move Cursor Up)
    3: "move_down",  # Fist Gesture (Move Cursor Down)
    6: "move_left",  # Index Gesture (Move Cursor Left)
    7: "move_right", # OK Gesture (Move Cursor Right)
    10: "click",     # Down Gesture (Simulate Click)
}

# Initialize previous gesture (for smooth control)
previous_gesture = None

# Start processing webcam frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize and normalize)
    frame_resized = cv2.resize(frame, (128, 128))  # Resize frame to match model input size
    frame_resized = frame_resized / 255.0  # Normalize the pixel values
    frame_input = frame_resized.reshape(1, 128, 128, 3)  # Add batch dimension

    # Predict gesture
    with torch.no_grad():
        outputs = model(torch.tensor(frame_input).float())
        _, predicted_class = torch.max(outputs, 1)
        predicted_class = predicted_class.item()

    # Handle the mouse movements and actions based on gestures
    if predicted_class == 1:  # Example: Palm Gesture (Move Up)
        pyautogui.move(0, -10)  # Move cursor up by 10 pixels
    elif predicted_class == 3:  # Fist Gesture (Move Down)
        pyautogui.move(0, 10)  # Move cursor down by 10 pixels
    elif predicted_class == 6:  # Index Gesture (Move Left)
        pyautogui.move(-10, 0)  # Move cursor left by 10 pixels
    elif predicted_class == 7:  # OK Gesture (Move Right)
        pyautogui.move(10, 0)  # Move cursor right by 10 pixels
    elif predicted_class == 10:  # Down Gesture (Click)
        pyautogui.click()  # Simulate a click

    # Display the frame with the predicted gesture (optional)
    cv2.putText(frame, f"Gesture: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture to Mouse", frame)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
