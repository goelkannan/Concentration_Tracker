# 🧠 Concentration Tracker

A real-time face analysis system that monitors **user concentration** using a webcam. It uses **MediaPipe FaceMesh**, **OpenCV**, and **NumPy** to track:

- 👁️ Eye Aspect Ratio (blink detection)
- 🎯 Gaze direction
- 🧭 Head pose alignment

The system then calculates a **concentration score**, shows it live on the video feed, and detects distraction over time.

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
git clone https://github.com/goelkannan/Concentration_Tracker.git
cd Concentration_Tracker
