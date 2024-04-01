from flask import Flask, request, redirect, url_for, render_template
from sklearn.discriminant_analysis import StandardScaler
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from joblib import load

# Ensure your feature extraction methods are correctly imported
from functions import (analyze_and_visualize_texture, analyze_and_visualize_frequency,
                       analyze_lbp_features, analyze_gabor_features, detect_face, aggregate_features,
                       predict_with_uncertainty, extract_face_frames)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure there's a folder for uploads
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('result', filename=filename))
    return render_template('upload.html')

def process_video(video_path):
    # Load the saved model
    clf = load('deepfake_detection_model.joblib')

    # Extract face frames from the video
    face_frames = extract_face_frames(video_path, max_frames=15)

    # Check if any face frames were extracted
    if not face_frames:
        return "no_face", None

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

@app.route('/analyze/<filename>')
def analyze(filename):
    return render_template('analyze.html', filename=filename)

@app.route('/result/<filename>')
def result(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result, confidence = process_video(video_path)
    if result == "no_face":
        return render_template('result.html', result="No face detected in the video.")
    else:
        confidence_percentage = f"{confidence:.2%}"
        print(f"Confidence Percentage: {confidence_percentage}")
        return render_template('result.html', result=result, confidence=confidence_percentage, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
