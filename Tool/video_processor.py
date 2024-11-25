import cv2

class VideoPreprocessor:
    def __init__(self, video_path):
        self.video_path = video_path

    def get_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames