from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the trained model
try:
    model = tf.keras.models.load_model('model.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500
        
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Get base64 string
        image_data = data['image']
        
        # Remove the header of the base64 string if present (e.g., "data:image/png;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        logging.info("Received image for prediction.")
        
        # Decode the image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        # 1. Convert to grayscale
        image = image.convert('L')
        # 2. Resize to 28x28
        image = image.resize((28, 28))
        
        # 3. Convert to numpy array
        img_array = np.array(image)
        logging.info(f"Image array shape before normalization: {img_array.shape}")
        
        # Note: the canvas has black background and white drawing (usually).
        # Our training data (MNIST) has black background and white digits.
        # But wait, default canvas is white and drawing is black unless specified.
        # We need to ensure the digit is white on a black background, matching MNIST.
        # If the web app canvas is white background with black pen:
        # We should invert the colors so background is 0 and digit is 255.
        
        # Let's assume the frontend sends a canvas with a black background and white stroke.
        # (We will implement the frontend that way).
        
        # 4. Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # 5. Reshape to (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # 6. Predict
        predictions = model.predict(img_array)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        logging.info(f"Predicted digit: {predicted_digit} with confidence {confidence:.4f}")
        
        return jsonify({
            'prediction': predicted_digit,
            'confidence': confidence
        })
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)