# import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification
import torch
from huggingface_hub import login
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from transformers import TFDistilBertModel, DistilBertTokenizerFast
from torch.nn import functional as F 

# tokenizer_sent = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
app = Flask(__name__)
CORS(
    app,
    supports_credentials=True
)

model_sent_name = "jm12312/Sentiment-text" 
# Load the model from Hugging Face
model_sent = DistilBertForSequenceClassification.from_pretrained(model_sent_name)

# Load the tokenizer
tokenizer_sent = DistilBertTokenizer.from_pretrained(model_sent_name)
# model_sent = joblib.load("models/sentiment/DistilbertFineTuned3_10krows_test.pkl")

@app.route('/predict/sentiment', methods=['POST'])
def predict_sentiment_score():
    try:
        data = request.json
        input_text = data.get('input')

        # Check if input text is provided
        if not input_text:
            return jsonify({'error': 'Input text is required'}), 400
        print(input_text)
        inputs = tokenizer_sent(input_text, padding=True, truncation=True, return_tensors='pt', max_length=128)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_sent.to(device)
        # Move input tensors to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model_sent(**inputs)
            predictions = outputs.logits.squeeze().cpu().numpy()  # Get the predicted scores

        positive_threshold = 0.1
        negative_threshold = 0.05

        # Print the predictions
        predictions = [predictions]
        op = 0
        for txt, score in zip([input_text], predictions):    
            print(f"Text: {txt} \nPredicted Sentiment: {score}\n")
            op = score
        # op = round(op, 3)
        # Return the sentiment score
        return jsonify({'sentiment_score': str(op)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500




model_emotion = joblib.load("models/emotion/emotion_model_test6_1050text.pkl")
tokenizer_emotion = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

dict1 = {0: "anger", 1: 'disgust', 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}

@app.route('/predict/emotion', methods=['POST'])
def predict_emotion_route_model():
    # Get the input from the user
    input_text = request.json.get("input", "")

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize the input text
    tokenized_input = tokenizer_emotion(input_text, padding=True, truncation=True, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model_emotion(**tokenized_input)

    # Get logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)

    # Get the predicted label
    prediction = torch.argmax(logits, dim=-1)

    # Convert probabilities and predictions to a list
    predicted_probabilities = probabilities.numpy().tolist()[0]
    predicted_label = prediction.numpy()[0]

    prob_str = {dict1[i]: round(predicted_probabilities[i], 4) for i in range(len(dict1))}

    result = {
        "input": input_text,
        "predicted_label": dict1[predicted_label],
        "probabilities": prob_str
    }

    return jsonify(result)






# from transformers import AutoTokenizer, TFDistilBertForSequenceClassification 
# import tensorflow as tf
# from flask import Flask, request, jsonify

# dict1 = {
#     0: 'Anger',
#     1: 'Fear',
#     2: 'Joy',
#     3: 'Love',
#     4: 'Sadness',
#     5: 'Surprise'
# }

# # Load the pre-trained model and tokenizer from Hugging Face
# model_emotion_name = "jm12312/Emotion-text-tf"

# # Load the model and tokenizer correctly
# model_emotion = TFDistilBertForSequenceClassification.from_pretrained(model_emotion_name)
# tokenizer_emotion = DistilBertTokenizer.from_pretrained(model_emotion_name)


# def predict_emotion(text):
#     # Tokenize the input text using the tokenizer
#     encoding = tokenizer_emotion(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    
#     # Make prediction using the model
#     logits = model_emotion(**encoding).logits
    
#     # Get the predicted label (index of the highest logit)
#     predicted_label = np.argmax(logits.numpy(), axis=1)[0]
    
#     # Convert the label index to its corresponding emotion
#     emotion = dict1[predicted_label]
    
#     return emotion

# # Define the route for prediction
# @app.route('/predict/emotion', methods=['POST'])
# def predict_emotion_route():
#     # Get the input text from the request
#     data = request.get_json()
#     text = data.get("input", "")
    
#     if text == "":
#         return jsonify({"error": "No text provided"}), 400
    
#     # Get the predicted label
#     label = predict_emotion(text)
    
#     # Return the prediction as a response
#     return jsonify({"predicted_label": label})





# Replace with your Hugging Face token
# token = "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"

# Log in using the Hugging Face token
# login(token)
model_name = "jm12312/MTTM_2" 
# Load the model from Hugging Face
model_hate = DistilBertForSequenceClassification.from_pretrained(model_name)

# Load the tokenizer
tokenizer_hate = DistilBertTokenizerFast.from_pretrained(model_name)

@app.route('/predict/hate', methods=['POST'])
def predict_hate():
    text = request.json.get("text").lower()
    inputs = tokenizer_hate(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Set the model to evaluation mode
    model_hate.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model_hate(**inputs)
        logits = outputs.logits

    # Get the predicted class (for classification)
    predicted_class = torch.argmax(logits, dim=-1).item()
    probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
    dic = {
        0: "Not hate",
        1: "Hate"
    }
    response = {
        "prediction": dic[predicted_class],
        "probabilities": {
            "Not hate": probabilities[0],
            "Hate": probabilities[1]
        }
    }

    return jsonify(response)







from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
from werkzeug.utils import secure_filename


# Set up a path for the uploaded files
UPLOAD_FOLDER = 'images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Replace with the actual Hugging Face model repository
model_repo = 'jm12312/HarmfulObjects'  # E.g., 'ultralytics/yolov8'
model_path = hf_hub_download(repo_id=model_repo, filename='best.pt')
model = YOLO(model_path)

@app.route('/predict/hate-image', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run YOLO model on the uploaded image
    results = model(file_path)

    # Prepare the response
    output = []
    for result in results:
        result.show()  # You can optionally save the image with bounding boxes
        result_data = {
            'boxes': [],
            'image_path': file_path,
            'detections': len(result.boxes)
        }

        # Loop through boxes in the result
        for box in result.boxes:
            box_data = {
                'class_id': box.cls.item(),
                'confidence': box.conf.item(),
                'class_probabilities': box.conf.softmax(dim=0).tolist()
            }
            result_data['boxes'].append(box_data)

        output.append(result_data)

    # Return the detection results as a JSON response
    return jsonify(output)
# import moviepy
# print(moviepy.__file__)
from supervision import Detections
from fer import FER
from PIL import Image
import contextlib
import os
import sys
import logging
import tempfile
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("fer").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Initialize models
with suppress_output():
    model_path = hf_hub_download(
        repo_id="arnabdhar/YOLOv8-Face-Detection",
        filename="model.pt"
    )
    model_yolo_emotions = YOLO(model_path)
    emotion_detector = FER(mtcnn=True)

# Emotion detection logic
def detect_emotions(image_path):
    try:
        with Image.open(image_path) as img:
            image = np.array(img.convert('RGB'))
        
        with suppress_output():
            results = model_yolo_emotions(image, verbose=False)
            detections = Detections.from_ultralytics(results[0])
        
        emotions = []
        for box in detections.xyxy:
            x_min, y_min, x_max, y_max = map(int, box[:4])
            face = image[y_min:y_max, x_min:x_max]
            
            with suppress_output():
                detected = emotion_detector.detect_emotions(face)
            
            if detected:
                emotion_scores = detected[0]["emotions"]
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                emotions.append(dominant_emotion)
        
        return emotions[0] if emotions else None
    except Exception as e:
        return None

# Flask route
@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image_path = temp_file.name
            image_file.save(image_path)
        
        emotion = detect_emotions(image_path)
        os.remove(image_path)

        if emotion:
            return jsonify({'emotion': emotion})
        else:
            return jsonify({'emotion': 'No face/emotion detected'}), 200
    except Exception as e:
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

import easyocr
import tempfile
reader = easyocr.Reader(['en'])
# OCR function
def apply_ocr(image_path):
    result = reader.readtext(image_path)
    extracted_text = " ".join([res[1] for res in result])
    return extracted_text

# Flask route for OCR
@app.route('/api/ocr', methods=['POST'])
def ocr_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image_path = temp_file.name
            image_file.save(image_path)

        extracted_text = apply_ocr(image_path)
        os.remove(image_path)

        return jsonify({'ocr_text': extracted_text})

    except Exception as e:
        return jsonify({'error': 'Server error', 'details': str(e)}), 500
if __name__ == '__main__':
        # Create the uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=8000, debug=True)