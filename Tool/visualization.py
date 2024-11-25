import cv2

class Visualizer:
    def visualize_results(self, frames, results, output_path):
        for i, (frame, (players, ball)) in enumerate(zip(frames, results)):
            for player in players:
                continue
            if ball:
                continue
            cv2.imwrite(f"{output_path}_frame_{i}.jpg", frame)