import os
import shutil
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)


UPLOAD_FOLDER = 'C:\\Jupyter\\Medical_Diagnosis\\static\\uploads'
NORMAL_FOLDER = 'C:\\Jupyter\\Medical_Diagnosis\\static\\uploads\\normal'
PNEUMONIA_FOLDER = 'C:\\Jupyter\\Medical_Diagnosis\\static\\uploads\\pneumonia'
CHART_FOLDER = 'C:\\Jupyter\\Medical_Diagnosis\\static\\images\\charts'
MODEL_PATH = 'C:\\Jupyter\\Medical_Diagnosis\\models\\pneumonia_model.pkl'

os.makedirs(NORMAL_FOLDER, exist_ok=True)
os.makedirs(PNEUMONIA_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)


model = joblib.load(MODEL_PATH)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = Image.open(filepath).resize((150, 150)).convert('L')
        img_array = np.array(img).flatten().reshape(1, -1)  

        prediction = model.predict(img_array)
        result = "Pneumonia" if prediction[0] == 1 else "Normal"

        target_folder = PNEUMONIA_FOLDER if result == "Pneumonia" else NORMAL_FOLDER
        unique_filename = file.filename
        while os.path.exists(os.path.join(target_folder, unique_filename)):
            name, ext = os.path.splitext(unique_filename)
            unique_filename = f"{name}_copy{ext}"
        shutil.move(filepath, os.path.join(target_folder, unique_filename))

        return render_template(
            'result.html',
            result=result,
            image_path=f"uploads/{'pneumonia' if result == 'Pneumonia' else 'normal'}/{unique_filename}"
        )

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/analytics')
def analytics():
    normal_count = len(os.listdir(NORMAL_FOLDER))
    pneumonia_count = len(os.listdir(PNEUMONIA_FOLDER))

    labels = ['Normal', 'Pneumonia']
    sizes = [normal_count, pneumonia_count]
    if sum(sizes) == 0:  
        sizes = [1, 1] 
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#FF5733'])
    plt.title('Diagnosis Summary')
    plt.savefig(os.path.join(CHART_FOLDER, 'diagnosis_chart.png'))
    plt.close()

    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True)
