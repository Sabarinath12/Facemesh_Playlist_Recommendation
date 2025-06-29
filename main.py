import cv2
import mediapipe as mp
import numpy as np
import webbrowser
from collections import deque
class EmotionMusicPlayer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.emotion_history = deque(maxlen=30)
        self.current_emotion = "neutral"
        self.music_config = {
            "happy": ["37i9dQZF1DXdPec7aLTmlC", "37i9dQZF1DX0XUsuxWHRQd", "37i9dQZF1DXcBWIGoYBM5M"],
            "sad": ["37i9dQZF1DWX83CujKHHOn", "37i9dQZF1DXdbXrPNafg9d", "37i9dQZF1DWVrtsSlLKzro"],
            "angry": ["37i9dQZF1DWWOaP4H0w5b0", "37i9dQZF1DX9qNs32fujYe", "37i9dQZF1DXcF6B6QPhFDv"],
            "surprised": ["37i9dQZF1DX0BcQWzuB7ZO", "37i9dQZF1DXcBWIGoYBM5M", "37i9dQZF1DX4JAvHpjipBk"]
        }
    def calculate_emotion(self, landmarks):
        if not landmarks:
            return "neutral"
        h, w = 480, 640
        points = np.array([[landmark.x * w, landmark.y * h] for landmark in landmarks.landmark])
        left_mouth, right_mouth, top_lip, bottom_lip = points[61], points[291], points[13], points[14]
        left_eye_top, left_eye_bottom, right_eye_top, right_eye_bottom = points[159], points[145], points[386], points[374]
        left_eyebrow, right_eyebrow = points[70], points[300]
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        mouth_width = abs(right_mouth[0] - left_mouth[0])
        lip_distance = abs(top_lip[1] - bottom_lip[1])
        avg_eye_openness = (abs(left_eye_top[1] - left_eye_bottom[1]) + abs(right_eye_top[1] - right_eye_bottom[1])) / 2
        eyebrow_height = (left_eyebrow[1] + right_eyebrow[1]) / 2
        mouth_curve_ratio = (mouth_center_y - top_lip[1]) / mouth_width if mouth_width > 0 else 0
        if mouth_curve_ratio < -0.02 and avg_eye_openness > 8:
            return "happy"
        elif mouth_curve_ratio > 0.02:
            return "sad"
        elif avg_eye_openness > 12:
            return "surprised"
        elif eyebrow_height < left_eye_top[1] - 10:
            return "angry"
        return "neutral"
    def smooth_emotion(self, emotion):
        self.emotion_history.append(emotion)
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        return max(emotion_counts, key=emotion_counts.get)
    def open_spotify_music(self, emotion):
        if emotion in self.music_config:
            playlist_id = np.random.choice(self.music_config[emotion])
            spotify_url = f"spotify:playlist:{playlist_id}"
            try:
                webbrowser.open(spotify_url)
            except:
                webbrowser.open(f"https://open.spotify.com/playlist/{playlist_id}")
    def draw_emotion_info(self, image, emotion, confidence):
        emotion_colors = {
            "happy": (0, 255, 0), "sad": (255, 0, 0), "angry": (0, 0, 255),
            "surprised": (0, 255, 255), "neutral": (128, 128, 128)
        }
        color = emotion_colors.get(emotion, (255, 255, 255))
        cv2.putText(image, f"Emotion: {emotion.upper()}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        bar_width = int(confidence * 200)
        cv2.rectangle(image, (10, 60), (10 + bar_width, 80), color, -1)
        cv2.rectangle(image, (10, 60), (210, 80), (255, 255, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            current_emotion = "neutral"
            confidence = 0.0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                    detected_emotion = self.calculate_emotion(face_landmarks)
                    smoothed_emotion = self.smooth_emotion(detected_emotion)
                    emotion_counts = {e: self.emotion_history.count(e) for e in self.emotion_history}
                    if smoothed_emotion in emotion_counts:
                        confidence = emotion_counts[smoothed_emotion] / len(self.emotion_history)
                    current_emotion = smoothed_emotion
                    if current_emotion != self.current_emotion and confidence > 0.6 and current_emotion != "neutral":
                        self.open_spotify_music(current_emotion)
                        self.current_emotion = current_emotion
            self.draw_emotion_info(frame, current_emotion, confidence)
            cv2.imshow('Emotion Music Player', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        player = EmotionMusicPlayer()
        player.run()
    except Exception as e:
        print(f"Error: {e}")
