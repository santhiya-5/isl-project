import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the path to your dataset
DATA_DIR = './data'  # Adjust this path according to where your dataset is located

data = []
labels = []

# Process each image in the dataset
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            img = cv2.imread(img_full_path)
            if img is None:
                continue  # Skip if the image is not loaded correctly

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = []
                    y_ = []
                    data_aux = []

                    # Extract and normalize landmarks
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y

                        x_.append(x)
                        y_.append(y)

                    # Append normalized landmarks to data_aux
                    for x, y in zip(x_, y_):
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    data.append(data_aux)
                    labels.append(dir_)

# Save the processed data and labels
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete and saved to data.pickle")
