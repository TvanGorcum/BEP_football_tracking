import cv2
from ultralytics import YOLO

def detect_players(input_path, output_path):
    model = YOLO("C:/Studie/BEP/BEP_Football_tracking/BEP_football_tracking/YOLO_weights/Lasts/epoch35.pt")
    # Define path to video file

    # Run inference on the source
    results = model(input_path, conf= 0.5, show_labels= False, save=True, project=output_path, name= 'test_result')
    print(results)

    return results

    #Put yolo model here that works on video or single frame
    #video
    #Use stream=True for processing long videos or large datasets to efficiently manage memory. When stream=False, the results for all frames or data points


def detect_ball(self, frame):
     # Placeholder for ball detection logic
    return self

detect_players(input_path='C:\Studie\BEP\BEP_Football_tracking/PSV_shaktar_demo_lines.mp4', output_path= 'C:/Studie/BEP/BEP_Football_tracking/results_tool')