from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Konfigürasyon
UPLOAD_FOLDER = 'uploadsOfApp'
RESULT_FOLDER = 'resultsOfApp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "runs/detect/train/weights/best.pt" 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Klasörleri oluştur
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULT_FOLDER).mkdir(exist_ok=True)

# YOLO modelini yükle
model = YOLO(MODEL_PATH)

calorie_dict = {
    'pizza': 285,
    'burger': 250,
    'friedpatato': 365,
    'nugget': 300,
    'cola': 150,
    'hotdog': 150,
    'onionring': 200
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    
    detected_items = []
    total_calories = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            class_name = model.names[int(c)]
            calories = calorie_dict.get(class_name, 0)
            detected_items.append({'name': class_name, 'calories': calories})
            total_calories += calories

            # Bounding box çizimi
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'{class_name}: {calories} kcal'
            cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    result_path = os.path.join(app.config['RESULT_FOLDER'], 'result_' + os.path.basename(image_path))
    cv2.imwrite(result_path, img)
    
    return detected_items, total_calories, result_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            detected_items, total_calories, result_path = predict_image(filepath)
            return jsonify({
                'detected_items': detected_items,
                'total_calories': total_calories,
                'result_image': os.path.basename(result_path)
            })
    return render_template('index.html')

@app.route('/results/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/recalculate', methods=['POST'])
def recalculate_calories():
    data = request.json
    total_calories = 0
    for item in data:
        calories_per_100g = calorie_dict.get(item['name'], 0)
        total_calories += (calories_per_100g * item['grams']) / 100
    return jsonify({'total_calories': round(total_calories, 2)})

if __name__ == '__main__':
    app.run(debug=True)