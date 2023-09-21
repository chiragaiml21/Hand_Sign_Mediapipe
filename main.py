import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # Use the default camera (you can specify a different camera if needed)

# Create a directory to save the collected data
if not os.path.exists("alphabets"):
    os.mkdir("alphabets")

# Define the alphabet signs you want to collect (e.g., A, B, C)
alphabet_signs = ["A", "B", "C"]  # Add more as needed

# Initialize a dictionary to keep track of the collected frames for each sign
collected_frames = {sign: [] for sign in alphabet_signs}

while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If hands are detected, draw landmarks and collect frames
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the hand sign (you may need to customize this part)
            # For simplicity, we assume each hand sign corresponds to a specific letter in the alphabet
            sign = "A"  # You should implement logic to determine the sign based on user input or gestures

            # Save the frame as a NumPy array to the corresponding alphabet list
            if sign in alphabet_signs and len(collected_frames[sign]) < 30:
                frame_array = np.array(frame)
                collected_frames[sign].append(frame_array)

    # Display the frame with landmarks
    cv2.imshow('Hand Sign Detection', frame)

    # Exit the loop when 'q' is pressed or all frames are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or all(len(frames) == 30 for frames in collected_frames.values()):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Now, 'collected_frames' dictionary contains NumPy arrays for each hand sign
