import os
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from flask import redirect


# Load CSV Data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load PyTorch Model
NUM_CLASSES = 39  # Adjust based on the number of classes
pytorch_model = CNN.CNN(NUM_CLASSES)
pytorch_model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
pytorch_model.eval()

# Function to Predict Using PyTorch Model
def predict_pytorch(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensures 3 channels
    image = image.resize((224, 224))
    input_tensor = TF.to_tensor(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
    output = pytorch_model(input_tensor).detach().numpy()
    index = np.argmax(output)
    return index

# Initialize Flask App
app = Flask(__name__)

# Routes
@app.route('/')
def home():
    return render_template('pages/home.html')

@app.route('/about')
def about():
    return render_template('pages/about.html')
@app.route('/services')
def services():
    return render_template('pages/service.html')

@app.route('/blog')
def blog():
    return render_template('pages/blog.html')

@app.route('/contact')
def contact():
    return render_template('pages/contact.html')

@app.route('/testimonial')
def testimonial():
    return render_template('pages/testimonial.html')

@app.route('/crop_information')
def crop_information():
    return render_template('pages/crop_information.html')

@app.route('/disease_infomation')
def disease_infomation():
    return render_template('pages/disease_info.html')

@app.route('/vegetable_info')
def vegetable_info():
    return render_template('pages/vegetable_info.html')


@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/product')
def product():
    return render_template('product.html')

# @app.route('/market')
# def market():
#     return render_template('market.html')


@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        upload_dir = os.path.join('static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        image.save(file_path)

        # Predict using PyTorch
        pred = predict_pytorch(file_path)

        # Get info from CSV
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

    return redirect('/index')  # Redirect to AI page if accessed via GET

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
