from gui import launch_gui
from video_processor import VideoPreprocessor
from keypoint_detection import KeypointDetector
from object_detection import PlayerDetector
from visualization import Visualizer
from tkinter import messagebox


def pixel_to_ground(P, x, z=0):
    """
    Converts a 2D image point to 3D ground plane coordinates.

    Parameters:
        P (numpy.ndarray): 3x4 camera projection matrix. From PNLcalib
        x (tuple): 2D pixel coordinates (u, v). from YOLO
        z (float): Height above the ground plane (default is 0).

    Returns:
        X, Y: Ground plane coordinates.
    """
    # 1. Split projection matrix into components
    P1, P2, P3 = P[0, :], P[1, :], P[2, :]

    # 2. Extract pixel coordinates
    u, v = x

    # 3. Solve for scaling factor lambda (λ)
    lambda_factor = -(P3[3] + P3[0] * u + P3[1] * v) / (P3[2] * z + P3[2])

    # 4. Solve for ground plane X and Y
    X = lambda_factor * (P1[3] + P1[0] * u + P1[1] * v)
    Y = lambda_factor * (P2[3] + P2[0] * u + P2[1] * v)

    return X, Y


# Example usage
P = np.array([[2.30691952e+06, -6.38192681e+05, 7.41301716e+04, 4.70977402e+07],  # Example camera projection matrix
              [-4.42593730e+04, 1.17789503e+05, 2.39140739e+06, 1.37569825e+07],
              [-2.67400385e-01, -9.62642642e-01, 4.26166307e-02, 7.82845187e+01]])

pixel_coords = (388, 592)  # Replace with bounding box center (u, v)

X, Y = pixel_to_ground(P, pixel_coords)
print(f"Ground coordinates: X={X}, Y={Y}, Z=0")
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
    playerdetections = PlayerDetector(input_path, output_path)
    print(playerdetections)




    # Visualization (if enabled)
    if visualize:
        visualizer = Visualizer()
        visualizer.visualize_results(frames, results, output_path)

    # Save results
    save_results(results, output_path)

if __name__ == "__main__":
    main()