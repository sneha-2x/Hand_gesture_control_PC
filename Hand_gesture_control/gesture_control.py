import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyautogui
import time

#  Load trained model
model = joblib.load("gesture_model.pkl")

#  Mediapipe setup 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

#  Webcam 
cap = cv2.VideoCapture(0)

#  Cooldown system 
last_trigger_time = {}
COOLDOWN = 1.5  # seconds

def can_trigger(label):
    current = time.time()
    if label not in last_trigger_time or (current - last_trigger_time[label] > COOLDOWN):
        last_trigger_time[label] = current
        return True
    return False

print(" Show gestures. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract features
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(data)[0]

            # Show label on screen
            cv2.putText(frame, prediction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            #  Trigger actions
            if prediction == "index_up":
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h)
                pyautogui.moveTo(x, y)

            elif prediction == "v_sign":
                pyautogui.click()

            elif prediction == "fist":
                pyautogui.rightClick()

            elif prediction == "thumbs_up" and can_trigger("thumbs_up"):
                pyautogui.press("volumeup")

            elif prediction == "thumbs_down" and can_trigger("thumbs_down"):
                pyautogui.press("volumedown")

            elif prediction == "open_palm" and can_trigger("open_palm"):
                pyautogui.press("playpause")

            elif prediction == "rock_sign" and can_trigger("rock_sign"):
                pyautogui.press("volumemute")

        

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
