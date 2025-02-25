import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.svm import OneClassSVM

# Video capture from webcam (Change 0 to a file path for video input)
cap = cv2.VideoCapture(0)

# Parameters
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)  # Background subtractor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Morphological kernel

# Initialize an empty list to hold feature vectors for ML training
motion_features = []

# Training an anomaly detection model using OneClassSVM
def train_model(features):
    if len(features) == 0:
        print("Error: Not enough motion features to train the model.")
        return None
    print("Training model with features shape:", np.array(features).shape)
    clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)  # Anomaly detection model
    clf.fit(features)  # Train with normal motion features
    return clf

# Feature extraction function
def extract_motion_features(contours):
    features = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small areas
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            features.append([aspect_ratio, area, perimeter])
    return np.array(features).reshape(-1, 3) if features else np.array([])

# Function to display or save frames
def show_or_save_frame(frame, fg_mask, frame_count):
    try:
        # Attempt to display the frames using OpenCV's imshow
        cv2.imshow('Motion Detection', frame)
        cv2.imshow('Foreground Mask', fg_mask)
    except cv2.error:
        # If in a headless environment, save images instead
        print(f"Saving frame {frame_count} as images...")
        cv2.imwrite(f'motion_detection_frame_{frame_count}.jpg', frame)
        cv2.imwrite(f'foreground_mask_{frame_count}.jpg', fg_mask)

        # Display using matplotlib for Jupyter environments
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 5))
        
        # Plot the motion detection frame
        plt.subplot(1, 2, 1)
        plt.imshow(frame_rgb)
        plt.title('Motion Detection Frame')

        # Plot the foreground mask
        plt.subplot(1, 2, 2)
        plt.imshow(fg_mask, cmap='gray')
        plt.title('Foreground Mask')

        plt.show()

# Function to detect anomalies
def detect_anomalies(clf, features):
    if len(features) == 0:
        return []
    predictions = clf.predict(features)
    print("Predictions:", predictions)  # Debugging output
    return predictions

# Main loop for capturing motion and anomaly detection
frame_count = 0
anomaly_clf = None  # Placeholder for anomaly detection model

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. Exiting...")
        break
    frame_count += 1

    # Pre-process the frame
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours (potential motions)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract features from contours for motion detection
    features = extract_motion_features(contours)

    # If enough features are collected, append to training data
    if len(features) > 0:
        motion_features.extend(features.tolist())

    # Train the model once enough data is collected (adjust threshold for better results)
    if len(motion_features) >= 200 and anomaly_clf is None:
        print("Training SVM for anomaly detection...")
        anomaly_clf = train_model(motion_features)

    # If the model is trained, check for anomalies in real-time
    anomalies = np.array([])  # Initialize anomalies to prevent NameError
    if anomaly_clf is not None:
        anomalies = detect_anomalies(anomaly_clf, features)

    # Draw bounding boxes for detected motion and highlight anomalies
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if i < len(anomalies) and anomalies[i] == -1:
                # Highlight anomaly in red
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                # Regular motion in green
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show or save frames based on the environment
    show_or_save_frame(frame, fg_mask, frame_count)

    # Simulate frame delay
    time.sleep(0.03)  # 30ms delay (approx. 30 fps)

    # Allow OpenCV to process key events
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User exited. Stopping...")
        break

    # Break loop after 300 frames (or any other stopping condition)
    if frame_count > 300:
        print("Max frames reached. Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
