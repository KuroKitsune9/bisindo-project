import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

DATA_PATH = "data/processed/dataset.csv"

os.makedirs("data/processed", exist_ok=True)

def save_data(label, landmarks):
    row = [label] + landmarks
    with open(DATA_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    if key == ord('a'):
        save_data("A", landmarks)
        print("Saved A")

    elif key == ord('b'):
        save_data("B", landmarks)
        print("Saved B")

    elif key == ord('c'):
        save_data("C", landmarks)
        print("Saved C")

    elif key == ord('d'):
        save_data("D", landmarks)
        print("Saved D")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()