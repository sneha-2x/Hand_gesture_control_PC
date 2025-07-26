# 🤚 Hand Gesture Control for PC using Machine Learning

Control your computer using only hand gestures captured through your webcam. This project uses MediaPipe, a trained Random Forest model, and PyAutoGUI to map real-time hand gestures to common PC actions like mouse movement, clicks, volume control, media playback, and navigation.

---

## 🧠 Features

| Gesture         | Fingers / Motion        | Action              |
|----------------|--------------------------|---------------------|
| `index_up`      | Index finger up           | Move mouse cursor   |
| `v_sign`        | Index + middle up         | Left click          |
| `fist`          | All fingers down          | Right click         |
| `thumbs_up`     | Thumb up                  | Volume up           |
| `thumbs_down`   | Thumb down                | Volume down         |
| `open_palm`     | All fingers open          | Play / Pause        |
| `rock_sign`     | Thumb + index + pinky     | Mute / Unmute       |
| `wave_left`     | Hand move left            | Next page / tab     |
| `wave_right`    | Hand move right           | Previous page / tab |

---

## 📂 Project Structure

Hand_gesture_control/
├── collect_gesture_data.py # Script to collect gesture data
├── train_gesture_model.py # Script to train Random Forest model
├── gesture_control.py # Real-time gesture control app
├── gesture_data.csv # Collected landmark data
├── gesture_model.pkl # Trained gesture classifier
├── README.md # You're here!

