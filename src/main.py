import os
from PIL import Image
from pre_load_img.ELA.convertimg import convert_to_ela
from pre_load_img.ELA.checktools import find_fake_real_files
from pre_load_img.PRNU.convert import extract_single, freqq, consis_map # Đã khôi phục
from glob import glob
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
from keras.models import load_model
from skimage.color import rgb2gray
import json
paras = {
    'noise_shape': (224, 224),
    'sigma': 1.5
}

with open('paras_model.json', 'w') as f:
    json.dump(paras, f)

def load_model_prnu(model_path, paras_path):    
    model = load_model(model_path)

    with open(paras_path, 'r') as f:
        paras = json.load(f)

    return model, paras

def load_model_ela(model_path):
    model = load_model(model_path)
    return model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_trained')
PARAS_PATH = os.path.join(MODEL_DIR, 'paras_model.json')
ELA_MODEL_PATH = os.path.join(MODEL_DIR, 'ela_model.h5')
PRNU_MODEL_PATH = os.path.join(MODEL_DIR, 'prnu_model.h5')

model_ela = load_model_ela(ELA_MODEL_PATH)
model_prnu, paras = load_model_prnu(PRNU_MODEL_PATH, PARAS_PATH)

def predict(img_path, model_ela , model_prnu, paras):
    #ELA
    img = Image.open(img_path)
    ela = convert_to_ela(img) 
    ela = np.array(ela)
    ela = resize(ela, (224, 224), preserve_range=True, anti_aliasing=True)
    ela = ela / 255.0
    ela = ela.reshape(1, 224, 224, 3)
    prediction_ELA = model_ela.predict(ela)[0][0]
    
    #PRNU
    img_np = np.array(img) 
    noise = extract_single(img_np, sigma=paras['sigma'])
    noise = resize(noise, paras['noise_shape'], preserve_range=True, anti_aliasing=True)
    fft = freqq(noise)
    consis = consis_map(noise, window_size=8)
    
    noise = noise.reshape(1, paras['noise_shape'][0], paras['noise_shape'][1], 1)
    fft = fft.reshape(1, paras['noise_shape'][0], paras['noise_shape'][1], 1)
    consis = consis.reshape(1, paras['noise_shape'][0], paras['noise_shape'][1], 1)
    
    prediction_PRNU = model_prnu.predict([noise, fft, consis])[0][0]

    return prediction_ELA, prediction_PRNU

def main():
    img_path = './uploads/RSA-2003-15vlw0m.jpg'
    prediction_ELA , prediction_PRNU = predict(img_path, model_ela , model_prnu, paras) 
    print(f"Kết quả dự đoán ELA: ", prediction_ELA)
    print(f"Kết quả dự đoán PRNU: ", prediction_PRNU)
   

    if prediction_ELA < 0.67 and prediction_PRNU >= 0.36:
        print("Ảnh có thể là ảnh được chụp bởi máy ảnh và chưa qua chỉnh sửa, cắt ghép!")
    elif prediction_ELA >= 0.67 and prediction_PRNU >= 0.36:
        print("Ảnh có thể là ảnh được chụp bởi máy ảnh nhưng đã qua chỉnh sửa, cắt ghép!")
    elif prediction_ELA >= 0.67 and prediction_PRNU < 0.36:
        print("Ảnh có thể là ảnh được tạo ra bởi AI!")
    else:
        print("Ảnh có thể là ảnh được tạo ra bởi AI hoặc đã qua chỉnh sửa nặng!")


if __name__ == "__main__":
    main()
