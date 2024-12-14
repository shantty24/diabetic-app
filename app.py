from flask_cors import CORS
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
# Load the pre-trained model
model = load_model('DR_base.h5')

# Define the image size
width = 150
height = 150

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of your model
    resized_image = image.resize((width, height))
    # Convert image to numpy array
    img_array = np.asarray(resized_image)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Expand dimensions to match the shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    # Example: Assuming the model returns class probabilities
    return jsonify({'predictions': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)