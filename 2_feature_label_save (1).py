import numpy as np
import os

def load_keypoints(keypoint_file_path):
    return np.load(keypoint_file_path)

def process_gesture(gesture, input_dir):
    gesture_dir = os.path.join(input_dir, gesture)
    gesture_features = []
    for keypoint_file in os.listdir(gesture_dir):
        keypoint_file_path = os.path.join(gesture_dir, keypoint_file)
        keypoints = load_keypoints(keypoint_file_path)
        gesture_features.append(keypoints)
    return gesture_features, [gesture] * len(gesture_features)

def save_features_and_labels(features, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    features = np.array(features)
    labels = np.array(labels)
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    print(features.shape, labels.shape)
    print("Features and labels have been saved.")

def main():
    input_dir = 'Extracted_Keypoints'
    output_dir = 'Feature_Labels'
    
    gesture_dirs = os.listdir(input_dir)
    features = []
    labels = []

    for gesture in gesture_dirs:
        gesture_features, gesture_labels = process_gesture(gesture, input_dir)
        features.extend(gesture_features)
        labels.extend(gesture_labels)

    save_features_and_labels(features, labels, output_dir)

if __name__ == "__main__":
    main()
