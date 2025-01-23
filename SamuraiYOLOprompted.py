from IPython import display
import ultralytics
from ultralytics import YOLO
import cv2
import os
import re
import glob
import shutil
import subprocess
framecounter = 0
frame = 1
seqlength = int
previousprompt = int # variable to check if the last prompt is the same as current prompt to avoid loop when samurai stops running at the first frame
#input your working directory
os.chdir("C:\Studie\BEP\BEP_Football_tracking\BEP_football_tracking\Samurai\samurai")
video_path = 'C:\Studie\BEP\BEP_Football_tracking\Dataset_ball_eval\sequences\SNMOT-137.mp4'
resolutionvideo = (1920, 1080) #Put resolution of video here(needed for YOLO to samurai annotation format conversion
output_dir = 'outputhybrid'
os.makedirs(output_dir, exist_ok=True)

originalseq = os.path.join(output_dir, 'originalseq')
os.makedirs(originalseq, exist_ok=True)

prompts = os.path.join(output_dir, 'prompts')
os.makedirs(prompts, exist_ok=True)

trackedseq = os.path.join(output_dir, 'trackedseq')
os.makedirs(trackedseq, exist_ok=True)

trackedseqtxt = os.path.join(output_dir, 'trackedseqtxt')
os.makedirs(trackedseqtxt, exist_ok=True)

YOLO_ball = os.path.join(output_dir, 'YOLO_ball')
os.makedirs(YOLO_ball, exist_ok=True)

originalseq = os.path.join(output_dir, 'originalseq')
os.makedirs(originalseq, exist_ok=True)

correctframes = os.path.join(output_dir, '../../correctframes')
os.makedirs(correctframes, exist_ok=True)

stopframes = os.path.join(output_dir, '../../stopframes')
os.makedirs(stopframes, exist_ok=True)
def convert_mp4_to_frames():
    """
    Converts an MP4 video into a folder of JPG frames.

    Args:
        video_path (str): Path to the input MP4 video.
        output_folder (str): Path to the folder where frames will be saved.
    """
    global originalseq
    global seqlength
    global video_path
    output_folder = originalseq

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 1

    while True:
        ret, frame = video_capture.read()

        # Break the loop when no frames are left to read
        if not ret:
            break

        # Define the frame's output filename
        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.jpg")

        # Save the current frame as a JPG file
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_count} to {frame_filename}")

        frame_count += 1
    seqlength = frame_count

    # Release the video capture object
    video_capture.release()
    print(f"Conversion complete! Total frames extracted: {frame_count}")

def convert_filename(yolo_file):
    # Use a regular expression to find the number at the end of the filename, before the extension
    match = re.search(r'(\d+)(?=\.[a-zA-Z]+$)', yolo_file)

    if match:
        # Extract the number (it can have multiple digits)
        number = match.group(1)

        # Return the new filename using only the extracted number and .txt extension
        return f"{number}.txt"
    else:
        raise ValueError("Filename does not have a number at the end")

def yolo_to_samurai(yolo_file, output_folder, resolution= resolutionvideo):
    # Define the target directory
    target_directory = output_folder #input of output_folder is global variable prompts

    # Convert YOLO filename to Samurai format filename
    output_filename = convert_filename(yolo_file)

    # Combine the target directory and the new filename to create the full path

    output_file = os.path.join(target_directory, output_filename)

    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)




    with open(yolo_file, 'r') as file:
        yolo_lines = file.readlines()

    samurai_lines = []
    for line in yolo_lines:
        # Split YOLO line into components
        values = line.split()
        if len(values) != 5:
            continue  # Skip malformed lines

        _, x_center_rel, y_center_rel, width_rel, height_rel = map(float, values)

        # Convert relative values to pixel values
        x_center = x_center_rel * resolution[0]
        y_center = y_center_rel * resolution[1]
        width = width_rel * resolution[0]
        height = height_rel * resolution[1]

        # Calculate top-left corner coordinates
        x_top_left = x_center - width / 2
        y_top_left = y_center - height / 2

        # Format as Samurai bounding box
        samurai_line = f"{int(round(x_top_left))}, {int(round(y_top_left))}, {int(round(width))}, {int(round(height))}\n"
        samurai_lines.append(samurai_line)

    # Write Samurai format to the specified output file path
    with open(output_file, 'w') as file:
        file.writelines(samurai_lines)

# Function that uses above two functions to process all the YOLO annotations into usable samurai prompts
def process_all_files(input_folder, output_folder): #input folder is YOLO_ball(and then a little more specific decide at yolo save output) Output folder is prompts
    # Find all .txt files in the directory
    yolo_files = glob.glob(os.path.join(input_folder, "*.txt"))

    for yolo_file in yolo_files:
        yolo_to_samurai(yolo_file, output_folder)

def copy_jpg_images(input_dir, output_dir): #input is global variabel originalseq and output is global variable trackedseq
    # Check if input directory exists

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input directory {input_dir} does not exist.")

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)

        # Only process files (skip directories)
        if os.path.isfile(input_file_path):
            # Check if the file is a .jpg file
            if filename.lower().endswith('.jpg'):
                # Create the output file path
                output_file_path = os.path.join(output_dir, filename)
                # Copy the .jpg file to the output directory
                shutil.copy(input_file_path, output_file_path)
                print(f"Copied: {filename}")


def find_initial_prompt(originalseq_path, prompts_path): #original seq is originalseq, prompts path is prompts

    global frame
    global samurai_running
    global framecounter

    # List and sort the text files in the prompts folder
    txt_files = sorted(os.listdir(prompts_path), key=lambda x: int(os.path.splitext(x)[0]))
    txt_frame_numbers = [int(os.path.splitext(file)[0]) for file in txt_files]

    while True:
        # Match frames with prompt text files
        if frame in txt_frame_numbers:
            samurai_running = True
            break
        else:
            # If the frame does not match, copy the corresponding image to trackedseqvis
            image_name = f"{frame:04d}.jpg"  # This assumes filename

            image_path = os.path.join(originalseq_path, image_name)

            if os.path.exists(image_path):

                # Remove the image from the original sequence
                os.remove(image_path)
            else:
                print(f"Image {image_name} not found in {originalseq_path}.")

        frame += 1
    framecounter = framecounter + frame




    print(f"YOLO Prompt found. Initializing Samurai: {samurai_running}")

def run_script(sequence_path, framecounter): #sequence path is originalseq framecounter is framecounter
    # Directory where the file exists
    global prompts

    directory = prompts

    # Generate the file name based on framecounter
    file_name = f"{framecounter}.txt"

    # Full path to the file
    file_path = os.path.join(directory, file_name)


    # Define the command with substitutions
    command = f'python scripts/demo.py --video_path {sequence_path} --txt_path {file_path}'

    # Run the command using subprocess
    subprocess.run(command, shell=True, check=True)

def move_sam_to_seq(input_folder, output_folder): #input is the correctframes variable, output is the trackedseqtxt folder
    """
    Processes text files in the input folder, renames them by adding the framecounter
    to the filename, copies them to the output folder, and clears the input folder.

    Args:
        input_folder (str): Path to the input directory.
        output_folder (str): Path to the output directory.
        framecounter (int): Value to add to the integer filenames.
    """
    global framecounter
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            # Extract the integer part of the filename
            try:
                base_name = int(os.path.splitext(filename)[0])
                # Add the framecounter to the integer part
                new_name = f"{base_name + framecounter -1}.txt"
                # Define full paths for input and output files
                input_file = os.path.join(input_folder, filename)
                output_file = os.path.join(output_folder, new_name)
                # Copy the file to the output folder with the new name
                shutil.copy(input_file, output_file)
            except ValueError:
                print(f"Skipping file '{filename}' as it does not have an integer name.")

    # Remove all files from the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print("Added samurai annotations to sequence and cleaned temporary samurai folder")

def get_stopframeno(directory): #directory input is stopframes variable
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    # If there are no .txt files, return None or raise an error
    if not txt_files:
        return None

    # Sort the list of files alphabetically (or by modification time if you prefer)
    txt_files.sort()  # Alphabetically

    # Get the last file
    last_file = txt_files[-1]

    # Remove the .txt extension and convert the result to an integer
    file_name_without_extension = last_file[:-4]


    try:
        return int(file_name_without_extension)
    except ValueError:
        # If the filename cannot be converted to an integer, return None
        return None

def find_nearest_higher_int_prompts(framecounter: int, folder_path: str) -> int: #folder path is prompts
    """
    Finds the nearest integer higher than `framecounter` from the filenames
    in the given folder.

    Args:
        framecounter (int): The reference integer.
        folder_path (str): The path to the folder containing text files with integer names.

    Returns:
        int: The nearest higher integer, or None if no such integer exists.
    """


    try:
        # List all files in the directory
        files = os.listdir(folder_path)

        # Extract integers from filenames
        integers = []
        for file in files:
            try:
                # Attempt to convert the filename (without extension) to an integer
                number = int(os.path.splitext(file)[0])
                integers.append(number)
            except ValueError:
                # Ignore files with non-integer names
                continue

        # Filter integers that are higher than framecounter
        higher_integers = [num for num in integers if num > framecounter]

        # Find the nearest higher integer
        if higher_integers:
            return min(higher_integers)
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def remove_frames_below_framecounter(folder_path, framecounter): #folder_path is originalseq, framecounter is framecounter
    """
    Remove all frames in the folder whose number is less than framecounter.

    Parameters:
        folder_path (str): Path to the folder containing the frames.
        framecounter (int): Threshold frame number. Files with numbers below this will be removed.
    """
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        for file in files:
            # Ensure the file has a .jpg extension
            if file.endswith('.jpg'):
                # Extract the numeric part of the filename
                try:
                    frame_number = int(file.split('.')[0])
                except ValueError:
                    # Skip files that don't follow the naming convention
                    continue

                # Check if the frame number is below the framecounter
                if frame_number < framecounter:
                    file_path = os.path.join(folder_path, file)
                    os.remove(file_path)  # Remove the file
                    print(f"Removed: {file}")

    except Exception as e:
        print(f"An error occurred: {e}")

def remove_all_files(directory):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")

#here for the actual portion of the code

convert_mp4_to_frames()
copy_jpg_images(originalseq, trackedseq)
model = YOLO("YOLOweight/best.pt")
results = model.predict(video_path, save=False, imgsz=1920, conf=0.50, show=True, save_txt= True, max_det = 1, project = 'outputhybrid/YOLO_ball')
process_all_files('outputhybrid/YOLO_ball/predict/labels', prompts)
find_initial_prompt(originalseq, prompts)
while framecounter <= seqlength:
framecounter = 1
run_script(originalseq, framecounter)
     move_sam_to_seq(correctframes, trackedseqtxt)

     frames_passed = get_stopframeno(stopframes) - 2
     framecounter = framecounter + frames_passed
     remove_all_files(stopframes)
     framecounter = find_nearest_higher_int_prompts(framecounter, prompts)

     remove_frames_below_framecounter(originalseq, framecounter)

#also #'ed out the code that stops running when box is bigger than certain amount of pixes



