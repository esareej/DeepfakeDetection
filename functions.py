from joblib import load
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from skimage.feature import local_binary_pattern

def analyze_and_visualize_texture(frame, visualize=False):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel edge detection filters in X and Y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine the Sobel X and Y results to get the overall edge magnitude
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize the result for visualization
    sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    
    if visualize:
        # Visualization
        plt.figure(figsize=(10, 5))
        
        # Original frame
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Frame')
        plt.axis('off')

        # Textural features (edges)
        plt.subplot(1, 2, 2)
        plt.imshow(sobel_normalized, cmap='gray')
        plt.title('Textural Features (Edges)')
        plt.axis('off')
        
        plt.savefig('static/graphs/texture_features.png')
        plt.close()
    
    # Returning the textural features for further processing or classification
    return sobel_normalized


def analyze_and_visualize_frequency(frame, visualize=False):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    
    if visualize:
        # Visualization
        plt.figure(figsize=(10, 5))
        
        # Original frame
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Original Frame')
        plt.axis('off')
        
        # Frequency domain
        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Frequency Domain')
        plt.axis('off')
        
        plt.savefig('static/graphs/frequency_features.png')  # Save the plot as an image
        plt.close()
    # Returning the frequency analysis for further processing or classification
    return magnitude_spectrum

def analyze_lbp_features(frame, visualize=False, P=8, R=1):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply LBP
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    if visualize:
        plt.figure(figsize=(5, 5))
        plt.plot(hist)
        plt.title('LBP Features')
        plt.xlabel('Bins')
        plt.ylabel('Normalized Frequency')
        plt.tight_layout()
        plt.savefig('static/graphs/lbp_features.png')  # Save the plot as an image
        plt.close()  # Close the plot to free up memory
    return hist

def analyze_gabor_features(frame, visualize=False, num=4):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Define Gabor filter parameters
    vecs = []
    filtered_images = []
    for theta in np.arange(0, np.pi, np.pi / num):
        gabor_kernel = cv2.getGaborKernel((21, 21), 8, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray, -1, gabor_kernel)
        vecs.append(np.mean(filtered))
        vecs.append(np.std(filtered))
        filtered_images.append(filtered)
    if visualize:
        fig, axes = plt.subplots(1, num, figsize=(20, 5))
        for i, img in enumerate(filtered_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Gabor Filter {i+1}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig('static/graphs/gabor_features.png')  # Save the plot as an image
        plt.close()  # Close the plot to free up memory
    return vecs

def extract_frames_from_folders(base_folder='data', analyze_texture=True, analyze_frequency=True, analyze_lbp=True, analyze_gabor=True, max_frames=15):
    categories = ['real', 'fake']
    data = {'real': [], 'fake': []}

    for category in categories:
        folder_path = os.path.join(base_folder, category)
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frames = []
            texture_features = []
            frequency_features = []
            lbp_features = []
            gabor_features = []
            frame_count = 0
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                face = detect_face(frame)
                if face is not None:
                    frames.append(face)
                    if analyze_texture:
                        texture_feature = analyze_and_visualize_texture(face, visualize=False)
                        texture_features.append(texture_feature)
                    if analyze_frequency:
                        frequency_feature = analyze_and_visualize_frequency(face, visualize=False)
                        frequency_features.append(frequency_feature)
                    if analyze_lbp:
                        lbp_feature = analyze_lbp_features(face, visualize=False)
                        lbp_features.append(lbp_feature)
                    if analyze_gabor:
                        gabor_feature = analyze_gabor_features(face, visualize=False)
                        gabor_features.append(gabor_feature)
                    frame_count += 1
            cap.release()
            data_entry = {'video_name': video_file, 'frames': frames}
            if analyze_texture:
                data_entry['texture_features'] = texture_features
            if analyze_frequency:
                data_entry['frequency_features'] = frequency_features
            if analyze_lbp:
                data_entry['lbp_features'] = lbp_features
            if analyze_gabor:
                data_entry['gabor_features'] = gabor_features
            data[category].append(data_entry)

    return data


def aggregate_and_visualize_features(data):
    # Initialize dictionaries to store aggregated features for each category
    aggregated_texture_features = {'real': [], 'fake': []}
    aggregated_frequency_features = {'real': [], 'fake': []}
    
    # Loop through each category and video
    for category, videos in data.items():
        for video in videos:
            # Aggregate texture features
            if video['texture_features']:
                texture_mean = np.mean([np.mean(feature) for feature in video['texture_features']])
                aggregated_texture_features[category].append(texture_mean)
            
            # Aggregate frequency features
            if video['frequency_features']:
                frequency_mean = np.mean([np.mean(feature) for feature in video['frequency_features']])
                aggregated_frequency_features[category].append(frequency_mean)
    
    # Visualization of aggregated texture features
    plt.figure(figsize=(10, 5))
    for category, features in aggregated_texture_features.items():
        if features:  # Check if there are features to plot
            plt.plot(features, label=f"{category} texture")
    plt.title('Aggregated Texture Features')
    plt.legend()
    plt.xlabel('Videos')
    plt.ylabel('Mean Texture Feature')
    plt.show()

    # Visualization of aggregated frequency features
    plt.figure(figsize=(10, 5))
    for category, features in aggregated_frequency_features.items():
        if features:  # Check if there are features to plot
            plt.plot(features, label=f"{category} frequency")
    plt.title('Aggregated Frequency Features')
    plt.legend()
    plt.xlabel('Videos')
    plt.ylabel('Mean Frequency Feature')
    plt.show()

# Example of calling the function with the data dictionary
# aggregate_and_visualize_features(data)

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        return frame[y:y+h, x:x+w]  # Return the first face found
    return None  # Return None if no faces are detected

def extract_face_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        face = detect_face(frame)
        if face is not None:
            frames.append(face)
            frame_count += 1
    cap.release()
    return frames

def aggregate_features(data):
    aggregated_features = []
    labels = []

    max_length = 0
    for category, videos in data.items():
        for video in videos:
            video_features = []
            if 'texture_features' in video and video['texture_features']:
                texture_mean = np.mean([np.mean(feat) for feat in video['texture_features']], axis=0)
                video_features.append(texture_mean)
            if 'frequency_features' in video and video['frequency_features']:
                frequency_mean = np.mean([np.mean(feat) for feat in video['frequency_features']], axis=0)
                video_features.append(frequency_mean)
            if 'lbp_features' in video and video['lbp_features']:
                lbp_mean = np.mean(np.array(video['lbp_features']), axis=0)
                video_features.extend(lbp_mean)
            if 'gabor_features' in video and video['gabor_features']:
                gabor_mean = np.mean(np.array(video['gabor_features']), axis=0)
                video_features.extend(gabor_mean)

            max_length = max(max_length, len(video_features))
            aggregated_features.append(video_features)

    # Pad the feature vectors to have the same length
    padded_features = []
    for features in aggregated_features:
        padded_features.append(np.pad(features, (0, max_length - len(features)), mode='constant'))

    for category in data:
        label = 1 if category == 'real' else 0
        labels.extend([label] * len(data[category]))

    return np.array(padded_features), np.array(labels)

def predict_with_uncertainty(clf, X_test, low_threshold=0.4, high_threshold=0.6):
    # Get predicted probabilities
    y_proba = clf.predict_proba(X_test)
    
    # Initialize an empty list for the modified predictions
    y_pred_modified = []
    
    for proba in y_proba:
        if proba.max() > high_threshold:  # High confidence
            y_pred_modified.append(proba.argmax())
        elif proba.max() >= low_threshold and proba.max() <= high_threshold:  # Unsure range
            y_pred_modified.append(2)  # Assuming 2 represents 'unsure'
        else:  # Moderate confidence but not unsure
            y_pred_modified.append(proba.argmax())
    
    return y_pred_modified

def process_video(video_path):
    # Load the saved model
    clf = load('deepfake_detection_model.joblib')

    # Extract face frames from the video
    face_frames = extract_face_frames(video_path, max_frames=15)

    # Check if any face frames were extracted
    if not face_frames:
        return "no_face", None

    # Select a specific frame to visualize its features
    selected_frame = face_frames[0]  # Choose the first frame, you can modify this as needed

    # Analyze and visualize features for the selected frame
    analyze_and_visualize_texture(selected_frame, visualize=True)
    
    analyze_and_visualize_frequency(selected_frame, visualize=True)
    
    analyze_lbp_features(selected_frame, visualize=True)

    analyze_gabor_features(selected_frame, visualize=True)


    # Initialize lists to store features
    texture_features = []
    frequency_features = []
    lbp_features = []
    gabor_features = []

    # Analyze and extract features from each face frame
    for frame in face_frames:
        texture_feature = analyze_and_visualize_texture(frame, visualize=False)
        texture_features.append(texture_feature)

        frequency_feature = analyze_and_visualize_frequency(frame, visualize=False)
        frequency_features.append(frequency_feature)

        lbp_feature = analyze_lbp_features(frame, visualize=False)
        lbp_features.append(lbp_feature)

        gabor_feature = analyze_gabor_features(frame, visualize=False)
        gabor_features.append(gabor_feature)

    # Create a data dictionary with the extracted features
    data = {
        'video': [
            {
                'texture_features': texture_features,
                'frequency_features': frequency_features,
                'lbp_features': lbp_features,
                'gabor_features': gabor_features
            }
        ]
    }

    # Aggregate the features
    aggregated_features, _ = aggregate_features(data)

    # Make predictions with uncertainty
    predictions = predict_with_uncertainty(clf, aggregated_features)

    # Determine the final prediction based on the majority
    final_prediction = max(set(predictions), key=predictions.count)

    if final_prediction == 0:
        result = 'fake'
    elif final_prediction == 1:
        result = 'real'
    else:
        result = 'unsure'

    # Calculate confidence
    confidence = clf.predict_proba(aggregated_features).max()
    print(f"Confidence: {confidence}")
    return result, confidence
