# 🧠 Concentration Tracker

A real-time face analysis system that monitors **user concentration** using a webcam. It uses **MediaPipe FaceMesh**, **OpenCV**, and **NumPy** to track:

- 👁️ Eye Aspect Ratio (blink detection)
- 🎯 Gaze direction
- 🧭 Head pose alignment

The system then calculates a **concentration score**, shows it live on the video feed, and detects distraction over time.

---

## 🎥 Demo

![demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXRpZ3l2ZDY0M3l6anRvbGV4aGV2N2tvbHJ0MDFodTJ4NmdyYjQ2bCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/hVnGU0NJnZHcAAcW2j/giphy.gif)

---

## 🛠️ Features

- 🔍 Detects blinks to infer focus loss
- 👀 Tracks gaze to check if you're looking at the screen
- 🧠 Computes a weighted concentration score in real-time
- 🟢 Shows concentration bar, blinking status, and distraction alerts
- ⚡ FPS meter and visual overlay

---

## 🧪 Tech Stack

- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [MediaPipe](https://pypi.org/project/mediapipe/)
- NumPy

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/concentration-tracker.git
cd concentration-tracker
