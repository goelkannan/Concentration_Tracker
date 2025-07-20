import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.python.solutions.drawing_utils import DrawingSpec

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

score_history = deque(maxlen=10)
distraction = 0

def eye_aspect_ratio(landmarks, eye_points, w, h, frame):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points]
    for x, y in points:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (A + B) / (2.0 * C)

def is_blinking(ear, threshold=0.2):
    return ear < threshold

def get_head_pose_score(landmarks, w, h):
    x, y = landmarks[1].x * w, landmarks[1].y * h
    d = np.linalg.norm([x - w / 2, y - h / 2])
    return 1.0 if d < 0.3 * w else 0.0

def get_gaze_score(landmarks):
    avg_x = (landmarks[468].x + landmarks[473].x) / 2.0
    return 1.0 if 0.5 < avg_x < 0.7 else 0.0

def compute_concentration_score(gaze, head_pose, blink):
    return round((0.4 * gaze + 0.4 * head_pose + 0.2 * (0 if blink else 1)) * 100, 2)

def draw_score_bar(frame, score):
    bar_x, bar_y, bar_w, bar_h = 30, 100, 200, 30
    fill = int(score * bar_w / 100)
    color = (0, 255, 0) if score > 40 else (0, 100, 255)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
    cv2.putText(frame, f"{score}%", (bar_x + bar_w + 10, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

cap = cv2.VideoCapture(0)
blink_counter, last_time = 0, time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    ui_overlay = frame.copy()
    cv2.rectangle(ui_overlay, (0, 0), (w, 150), (30, 30, 30), -1)
    frame = cv2.addWeighted(ui_overlay, 0.6, frame, 0.4, 0)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            lm = landmarks.landmark
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=DrawingSpec(color=(0, 150, 255), thickness=1)
            )

            ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h, frame) + 
                   eye_aspect_ratio(lm, RIGHT_EYE, w, h, frame)) / 2
            blink = is_blinking(ear)
            gaze = get_gaze_score(lm)
            head = get_head_pose_score(lm, w, h)
            concentration = compute_concentration_score(gaze, head, blink)

            score_history.append(concentration)
            smooth_score = int(np.mean(score_history))
            draw_score_bar(frame, smooth_score)

            cv2.putText(frame, f"Concentration: {smooth_score}%", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if blink:
                cv2.putText(frame, "BLINKING", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)

            if smooth_score < 40:
                distraction += 1
                cv2.putText(frame, f"Distraction: {distraction}", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                if distraction > 1000:
                    distraction = 0
                    print("Turn off trigger")

    # FPS Display using time delta
    current_time = time.time()
    fps = 1 / (current_time - last_time + 1e-6)
    last_time = current_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    status_color = (0, 255, 0) if distraction == 0 else (0, 100, 255)
    cv2.circle(frame, (w - 30, 70), 15, status_color, -1)
    cv2.putText(frame, "ACTIVE" if distraction == 0 else "DISTRACTED", 
                (w - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
