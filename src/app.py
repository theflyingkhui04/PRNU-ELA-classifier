from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
from main import model_ela, model_prnu, paras, predict 
from pre_load_img.ELA.convertimg import convert_to_ela
import numpy as np
from skimage.transform import resize

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    analysis_type = request.form.get('analysis_type', 'combined') 

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        img = Image.open(file)
        img_rgb = img.convert('RGB')
        img_resized = img_rgb.resize((224, 224)) 
        img_resized.save(filepath) 

        prediction_ELA, prediction_PRNU = predict(filepath, model_ela, model_prnu, paras)
        
        print(f"prediction_ELA: {prediction_ELA}, prediction_PRNU: {prediction_PRNU}, Analysis Type: {analysis_type}")

        result = ""
       
        if analysis_type == 'original_vs_edited':
            if prediction_ELA < 0.67:
                result = "Ảnh có thể là ảnh gốc, chưa qua chỉnh sửa."
            else:
                result = "Ảnh có thể là ảnh đã qua chỉnh sửa."
        elif analysis_type == 'real_vs_ai':
            
            if prediction_PRNU >= 0.36:
                result = "Ảnh có thể là ảnh thật (chụp từ camera)."
            else:
                result = "Ảnh có thể là ảnh tạo bởi AI."
        elif analysis_type == 'combined':
            if prediction_ELA < 0.67 and prediction_PRNU >= 0.36:
                result = "Ảnh có thể là ảnh thật và chưa qua chỉnh sửa."
            elif prediction_ELA >= 0.67 and prediction_PRNU >= 0.36:
                result = "Ảnh có thể là ảnh thật nhưng đã qua chỉnh sửa."
            elif prediction_PRNU < 0.36:
                result = "Ảnh có thể là ảnh tạo bởi AI."
            else: 
                result = "Không thể xác định rõ, ảnh có thể đã qua chỉnh sửa nặng hoặc tạo bởi AI."
        else:
            return jsonify({'error': 'Loại phân tích không hợp lệ'})

        return jsonify({'result': result, 'filename': filename, 'prediction_ELA': float(prediction_ELA), 'prediction_PRNU': float(prediction_PRNU)})
    return jsonify({'error': 'File không hợp lệ'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return app.send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
