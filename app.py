import base64
import logging
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize the Flask app
app = Flask(__name__)

# Load the VGG16 model and other relevant variables
# Replace this with the path to your model file
model = load_model('construction.h5')
img_width, img_height = 224, 224
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Function to preprocess and extract features from an image
def extract_features_from_image(image):
    img = Image.open(BytesIO(base64.b64decode(image)))
    img = img.resize((img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    features = conv_base.predict(img_array)
    return features

# Function to make predictions on an image
def make_prediction(image):
    features = extract_features_from_image(image)
    prediction = model.predict(features)
    logging.info('Image received')
    return prediction[0][0]

# Endpoint for making predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data.get('image_data')
    if image_data is None:
        return jsonify({'error': 'Invalid request: missing "image_data" key'}), 400

    prediction = make_prediction(image_data)
    result = {'prediction': 'YES' if prediction < 0.5 else 'NO'}
    logging.info('Predicted: %s', result['prediction'])
    return jsonify(result)

# New route to verify that the app is running
@app.route('/hello')
def hello():
    return "hello"

# Configure logging
logging.basicConfig(level=logging.INFO)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    
