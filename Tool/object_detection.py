import cv2
from ultralytics import YOLO

def detect_players(input_path, output_path):
    model = YOLO("C:/Studie/BEP/BEP_Football_tracking/BEP_football_tracking/YOLO_weights/players11/last.pt")
    # Define path to video file
    model1 = YOLO("yolo11n.pt")

    # Run inference on the source
    results = model(input_path, conf= 0.3, show_labels= False, save=True, project=output_path, name= 'test_result_players')
    print(results)

    return results

    #Put yolo model here that works on video or single frame
    #video
    #Use stream=True for processing long videos or large datasets to efficiently manage memory. When stream=False, the results for all frames or data points


def detect_ball(input_path, output_path):
     # Placeholder for ball detection logic
     modelball = YOLO("C:/Studie/BEP/BEP_Football_tracking/BEP_football_tracking/YOLO_weights/ball_11/best.pt")
     results = modelball(input_path, conf=0.5, show_labels=False, save=True, project=output_path, name='test_result_ball')
     return results

#detect_players(input_path='C:/Studie/BEP/BEP_Football_tracking/sample_nl_match.mp4', output_path= 'C:/Studie/BEP/BEP_Football_tracking/results_tool')
detect_ball(input_path='C:/Studie/BEP/BEP_Football_tracking/sample_nl_match.mp4' , output_path= 'C:/Studie/BEP/BEP_Football_tracking/results_tool')