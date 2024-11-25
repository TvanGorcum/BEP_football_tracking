from gui import launch_gui
from video_processor import VideoPreprocessor
from keypoint_detection import KeypointDetector
from object_detection import PlayerDetector
from visualization import Visualizer
from tkinter import messagebox

def main():
    # Launch GUI to collect inputs
    input_path, output_path, visualize = launch_gui()
    if input_path == '':
        messagebox.showerror("Error", "Input path was not specified")

    if output_path == '':
        messagebox.showerror("Error", "Output path was not specified")

    # Preprocess video into frames
    preprocessor = VideoPreprocessor(input_path)
    frames = preprocessor.get_frames()
    save_results(results, output_path)




    # Keypoint detection and homography
    keypoint_detector = KeypointDetector()
    homography_matrix = keypoint_detector.detect_keypoints_and_homography(frames[0])  # Use first frame for pitch detection

    # Player and ball detection
    player_detector = PlayerDetector(homography_matrix)
    results = []
    for frame in frames:
        continue
        #players, ball = player_detector.detect_players_and_ball(frame)
        #results.append((players, ball))


    # Visualization (if enabled)
    if visualize:
        visualizer = Visualizer()
        visualizer.visualize_results(frames, results, output_path)

    # Save results
    save_results(results, output_path)

if __name__ == "__main__":
    main()