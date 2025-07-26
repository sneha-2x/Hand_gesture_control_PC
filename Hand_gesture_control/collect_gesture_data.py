import cv2
import mediapipe as mp
import csv
import os

gesture_label = "wave_right" # This is the gesture name
output_csv = "gesture_data.csv"
samples_to_collect = 300

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
data = []
sample_count = 0

print(f"[INFO] Starting data collection for gesture: '{gesture_label}'")

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            row = [gesture_label] + landmarks
            data.append(row)
            sample_count += 1

    cv2.putText(frame, f"Samples: {sample_count}/{samples_to_collect}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting Gesture Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= samples_to_collect:
        break

cap.release()
cv2.destroyAllWindows()

file_exists = os.path.exists(output_csv)
with open(output_csv, 'a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        header = ["label"] + [f"{i}_{coord}" for i in range(21) for coord in ("x", "y", "z")]
        writer.writerow(header)
    writer.writerows(data)

print(f"[DONE] Collected {sample_count} samples for gesture '{gesture_label}' and saved to {output_csv}")
