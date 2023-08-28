

import mediapipe as mp
import cv2
import numpy as np
import os
from imgaug import augmenters as iaa

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Initialize Mediapipe Solutions
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Path to your dataset directory
data_dir = 'C:\\Users\\lapde\\OneDrive - Solent University\\Desktop\\my_final_codes\\Dataset'
output_dir = 'Extracted_Keypoints'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

gestures = os.listdir(data_dir)

# Temporal Augmentation setup
num_frames_per_sample = 30

# Image Augmentation setup
augmentation_seq = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Multiply((0.7, 1.3)),
    iaa.AddToHueAndSaturation((-20, 20)),
    iaa.ContrastNormalization((0.5, 2.0)),
    iaa.AverageBlur(k=(2, 7)),  # Adding average blurring
])

for gesture in gestures:
    print(gesture)
    gesture_dir = os.path.join(data_dir, gesture)
    gesture_output_dir = os.path.join(output_dir, gesture.replace("_Raw", ""))
    
    # Create gesture directory inside output directory
    if not os.path.exists(gesture_output_dir):
        os.makedirs(gesture_output_dir)
    
    for video_file in os.listdir(gesture_dir):
        video_path = os.path.join(gesture_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0

        colored_keypoints_list = []
        grayscale_keypoints_list = []

        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count} of {video_file}")
            
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize frames to a common size
            target_height, target_width = 240, 320
            resized_colored_frame = cv2.resize(frame, (target_width, target_height))
            resized_grayscale_frame = cv2.resize(gray_frame, (target_width, target_height))
            
            # Apply augmentation to colored frame
            augmented_colored_frame = augmentation_seq.augment_image(resized_colored_frame)
            
            # Process colored frame
            colored_image, results = mediapipe_detection(augmented_colored_frame, holistic)
            colored_keypoints = extract_keypoints(results)
            colored_keypoints_list.append(colored_keypoints)

            # Process grayscale frame
            grayscale_image, results = mediapipe_detection(resized_grayscale_frame, holistic)
            grayscale_keypoints = extract_keypoints(results)
            grayscale_keypoints_list.append(grayscale_keypoints)

            # Save keypoints as numpy files every num_frames_per_sample frames
            if len(colored_keypoints_list) == num_frames_per_sample:
                # Save colored keypoints
                colored_keypoint_file_path = os.path.join(gesture_output_dir, f"colored_{os.path.splitext(video_file)[0]}_{frame_count}_keypoints.npy")
                np.save(colored_keypoint_file_path, colored_keypoints_list)
                
                # Save grayscale keypoints
                grayscale_keypoint_file_path = os.path.join(gesture_output_dir, f"grayscale_{os.path.splitext(video_file)[0]}_{frame_count}_keypoints.npy")
                np.save(grayscale_keypoint_file_path, grayscale_keypoints_list)
                
                # Clear keypoints lists
                colored_keypoints_list = []
                grayscale_keypoints_list = []

        cap.release()

print("Processing complete.")
