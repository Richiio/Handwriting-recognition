from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

def recognize_text(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Load your model
    model = tf.keras.models.load_model('your_text_recognition_model_checkpoint.h5')

    # Perform text recognition
    predictions = model.predict(image)
    # Process predictions and extract text (replace this with your model's logic)
    # For example, if your model outputs probabilities for each character:
    text = ""
    for prediction in predictions:
        # Assuming prediction is a list of probabilities for each character
        # Convert probabilities to characters using a mapping
        predicted_char_index = np.argmax(prediction)
        predicted_char = chr(predicted_char_index + ord('a'))  # Example: Map indices to characters
        text += predicted_char
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        # Save the uploaded file temporarily
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        # Call the text recognition function
        extracted_text = recognize_text(image_path)
        os.remove(image_path)  # Remove the uploaded image after recognition
        return render_template('index.html', extracted_text=extracted_text)
    return "Error"

if __name__ == '__main__':
    app.run(debug=True)
