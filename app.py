from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from functions import process_video

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about_us.html')

@app.route('/terms')
def terms():
    return render_template('terms_of_service.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy_policy.html')

@app.route('/thank-you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Process the form submission
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # You can perform further actions with the form data here
        return redirect(url_for('thank_you'))
    return render_template('contact_us.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Handle the file upload and analysis
    file = request.files['file']
    
    # Save the uploaded video file
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    # Process the video and get the result and confidence
    result, confidence = process_video(video_path)
    
    if result == "no_face":
        # Display a pop-up message using JavaScript's alert()
        message = "No face was detected in the uploaded video. Please upload a new video with a clearer face."
        return f"<script>alert('{message}'); window.history.back();</script>"
    
    # Redirect to the result page with the analysis result and confidence
    return redirect(url_for('result', result=result, confidence=confidence))

@app.route('/result')
def result():
    result = request.args.get('result')
    confidence = float(request.args.get('confidence', 0))
    return render_template('result.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)