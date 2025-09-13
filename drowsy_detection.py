# drowsy_detection.py
import cv2
import mediapipe as mp
import numpy as np

class DrowsyDetector:
    def __init__(self, ear_thresh=0.18, wait_time=1.0):
        self.ear_thresh = ear_thresh
        self.wait_time = wait_time
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
        self.drowsy_start = None

    def get_ear(self, landmarks, refer_idxs, frame_w, frame_h):
        coords = []
        for idx in refer_idxs:
            lm = landmarks[idx]
            coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
        P2_P6 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        P3_P5 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        P1_P4 = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
        return ear

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        ear = 1.0
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = self.get_ear(landmarks, self.eye_idxs["left"], w, h)
            right_ear = self.get_ear(landmarks, self.eye_idxs["right"], w, h)
            ear = (left_ear + right_ear) / 2.0
        return ear
