import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils


#Path for exported data, numpy arrays
DATA_PATH = os.path.join('Alphabets')

#Actions that we try to detect
actions = []

#30 videos worth of data
no_sequence = 30

#Videos are going to be 30 frames in length
sequence_length = 30


for root, dirs, files in os.walk(DATA_PATH):
    for dir_name in dirs:
        if not dir_name.isdigit():
            actions.append(dir_name)

actions = np.array(actions)
print(actions)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('D://Study Material//SIH 2023//Hand_Sign_MediaPipe//models//alphabets.h5')



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # Draw left hand connections
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Determine if it's the left hand or right hand based on landmark positions
            if landmarks.landmark[mp_hands.HandLandmark.WRIST].x < landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
                # Left hand
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                           )
            else:
                # Right hand
                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                           )

    

def extract_keypoints(results):
    # Initialize empty arrays for left and right hand landmarks
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    # Check if multi_hand_landmarks are available
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Determine if it's the left hand or right hand based on landmark positions
            if landmarks.landmark[mp_hands.HandLandmark.WRIST].x < landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x:
                # Left hand
                lh = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()
            else:
                # Right hand
                rh = np.array([[res.x, res.y, res.z] for res in landmarks.landmark]).flatten()

    return np.concatenate([lh, rh])



# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # # Viz probabilities
            # image = prob_viz(res, actions, image, colors)
           
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()