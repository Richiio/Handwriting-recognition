# from flask import Flask, render_template, request
# import os
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# app = Flask(__name__)

# def recognize_text(image_path):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)

#     # Load your model
#     model = tf.keras.models.load_model('model.h5')

#     # Perform text recognition
#     predictions = model.predict(image)
#     text = ""
#     for prediction in predictions:
#         predicted_char_index = np.argmax(prediction)
#         predicted_char = chr(predicted_char_index + ord('a'))  # Example: Map indices to characters
#         text += predicted_char
#     return text

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file part"
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file"
#     if file:
#         # Save the uploaded file temporarily
#         image_path = os.path.join('uploads', file.filename)
#         file.save(image_path)
#         # Call the text recognition function
#         extracted_text = recognize_text(image_path)
#         os.remove(image_path)  # Remove the uploaded image after recognition
#         return render_template('index.html', extracted_text=extracted_text)
#     return "Error"

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, request
# import os
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# app = Flask(__name__)

# # Load your text recognition model
# def recognize_text(image_path):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert('L')  # Convert to grayscale
#     image = image.resize((32, 32))  # Resize to the input size of the model
#     image = np.array(image) / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     image = np.expand_dims(image, axis=-1)  # Add channel dimension (for grayscale)

#     # Load your model
#     model = tf.keras.models.load_model('model.h5')

#     # Perform text recognition
#     predictions = model.predict(image)
#     # Process predictions and extract text (replace this with your model's logic)
#     return predictions

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return "No file part"
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file"
#     if file:
#         # Save the uploaded file temporarily
#         image_path = os.path.join('uploads', file.filename)
#         file.save(image_path)
#         # Call the text recognition function
#         predictions = recognize_text(image_path)
#         os.remove(image_path)  # Remove the uploaded image after recognition
#         return render_template('index.html', predictions=predictions)
#     return "Error"

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your text recognition model
def recognize_text(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to the input size of the model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (for grayscale)

    # Load your model
    model = tf.keras.models.load_model('model.h5')

    # Perform text recognition
    predictions = model.predict(image)
    # Process predictions and extract text (replace this with your model's logic)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Save the uploaded file temporarily
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        # Call the text recognition function
        predictions = recognize_text(image_path)
        os.remove(image_path)  # Remove the uploaded image after recognition
        # Return predictions in a dictionary format
        prediction_dict = {'predictions': predictions.tolist()}
        return jsonify(prediction_dict)
    return jsonify({'error': 'Error'})

if __name__ == '__main__':
    app.run(debug=True)
