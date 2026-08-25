from flask import Flask, render_template, jsonify, send_file
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from gtts import gTTS
import os

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Labels dictionary
labels_dict = {0: 'Bye', 1: 'Yes', 2: 'Hi'}

# Function to capture ISL translation
def predict_gesture():
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    gesture_queue = deque(maxlen=10)
    most_common_gesture = "No gesture"

    while True:  # Keep the loop running to capture multiple gestures
        ret, frame = cap.read()
        if not ret:
            continue  # Skip to the next iteration if no frame is captured

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                # Extract hand landmarks and normalize coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Predict gesture based on normalized landmark coordinates
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                print(f"Predicted Gesture: {predicted_character}")  # Debug: Log predicted gestures

                gesture_queue.append(predicted_character)
                print(f"Gesture Queue: {gesture_queue}")  # Debug: Log the queue

        # Find the most common gesture in the queue
        if gesture_queue:
            most_common_gesture = max(set(gesture_queue), key=gesture_queue.count)
            print(f"Most Common Gesture: {most_common_gesture}")  # Debug: Log the most common gesture

            # If a valid gesture is detected, break the loop and return it
            if most_common_gesture != "No gesture":
                cap.release()  # Release the camera after detecting the gesture
                return most_common_gesture

# Function to convert text (gesture) to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    speech_file = "gesture_speech.mp3"
    tts.save(speech_file)
    return speech_file

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start_translation')
def start_translation():
    return render_template('start_translation.html')

@app.route('/get-gesture', methods=['GET'])
def get_gesture():
    gesture = predict_gesture()  # Keep predicting gestures
    return jsonify({'gesture': gesture})

@app.route('/get-speech/<gesture>', methods=['GET'])
def get_speech(gesture):
    if gesture != "No gesture":
        speech_file = text_to_speech(gesture)  # Convert the gesture to speech
    return send_file('gesture_speech.mp3')

if __name__ == '__main__':
    app.run(debug=True)
