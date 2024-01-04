from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
from flask_cors import CORS
from datetime import datetime
from inference import predict_image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Enable CORS for all routes

def extract_confidence(file_name):
    # Assuming the confidence is encoded in the file name, extract it
    try:
        confidence_str = file_name.split('_')[-1].split('.')[0]
        confidence = float(confidence_str)
        return confidence
    except (ValueError, IndexError):
        return 0.0  # Default confidence if extraction fails


def select_highest_confidence_image(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    if not files:
        return None  # No image files found

    # Extract confidence levels from file names
    confidence_levels = [extract_confidence(file) for file in files]

    # Find the index of the image with the highest confidence
    max_confidence_index = confidence_levels.index(max(confidence_levels))

    # Get the path of the image with the highest confidence
    highest_confidence_image = os.path.join(
        folder_path, files[max_confidence_index])

    return highest_confidence_image

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Check if the POST request has a file part

    file = request.files['file']

    # Check if the file is one of the allowed types/extensions
    # allowed_extensions = {'jpg', 'jpeg', 'png'}
    # if file.filename.split('.')[-1].lower() not in allowed_extensions:
    #     return render_template('error.html', error='Invalid file type')

    # Save the uploaded image to a temporary location
    upload_folder = 'Upload'
    if os.path.exists(upload_folder):
        # Remove the destination folder and its contents
        for file_ in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, file_)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(upload_folder)
    filename = file.filename
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    weights_path = "Yolo_Weights.pt"

    # Run YOLOv5 detection command
    detection_command = f'python yolov7-main/detect.py --weights "{weights_path}" --source "{file_path}" --save-cropped --conf 0.2'
    os.system(detection_command)
    cropped_folder = 'ResnetInput'
    selected_image_path = select_highest_confidence_image(cropped_folder)

    if not selected_image_path:
        files = os.listdir('Upload')
        upload_file_name = files[0]
        upload_file_path = os.path.join('Upload', upload_file_name)
        selected_image_path = upload_file_path
    final_prediction = predict_image(selected_image_path)
    print(final_prediction)
    
    # Return response to frontend 
    return render_template('index.html', final_predictions=str(final_prediction['predicted_class']),filename='uploads'+filename)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# final_predictions=final_prediction)
if __name__ == '__main__':
    app.run(debug=True, port=5001)
