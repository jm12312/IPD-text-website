# import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, DistilBertTokenizer
import torch
# tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# tokenizer_sent = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
app = Flask(__name__)
CORS(
    app,
    resources={r"/predict/*": {
        "origins": "http://localhost:3000"
        }
    }
)

# model_sent = joblib.load("models/sentiment/DistilbertFineTuned3_10krows_test.pkl")

# @app.route('/predict/sentiment', methods=['POST'])
# def predict_sentiment_score():
#     try:
#         data = request.json
#         input_text = data.get('input')

#         # Check if input text is provided
#         if not input_text:
#             return jsonify({'error': 'Input text is required'}), 400
#         print(input_text)
#         inputs = tokenizer_sent(input_text, padding=True, truncation=True, return_tensors='pt', max_length=128)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model_sent.to(device)
#         # Move input tensors to device
#         inputs = {key: value.to(device) for key, value in inputs.items()}

#         # Get predictions
#         with torch.no_grad():
#             outputs = model_sent(**inputs)
#             predictions = outputs.logits.squeeze().cpu().numpy()  # Get the predicted scores

#         positive_threshold = 0.1
#         negative_threshold = 0.05

#         # Print the predictions
#         predictions = [predictions]
#         op = 0
#         for txt, score in zip([input_text], predictions):    
#             print(f"Text: {txt} \nPredicted Sentiment: {score}\n")
#             op = score
#         # Return the sentiment score
#         return jsonify({'sentiment_score': str(op)})

#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({'error': str(e)}), 500

# dict1 = {0: "anger", 1: "disgust", 2: "fear", 3: "joy", 4: "neutral", 5: "sadness", 6: "surprise"}
# model_emotions = joblib.load("models/emotion/emotion_model_test6_1050text.pkl")
# @app.route('/predict/emotion', methods=['POST'])
# def predict_emotion():
#     try:
#         # Get input data from the request
#         data = request.json
#         input_text = data.get('input')

#         # Check if input text is provided
#         if not input_text or input_text == '':
#             return jsonify({'error': 'Input text is required'}), 400
        
#         # Tokenize the input text
#         tokenized_test_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

#         # Run the model in inference mode (no gradients)
#         with torch.no_grad():
#             outputs = model_emotions(**tokenized_test_inputs)

#         # Get logits from the model outputs
#         logits = outputs.logits

#         # Apply softmax to get probabilities for each emotion class
#         probabilities = F.softmax(logits, dim=-1)

#         # Get the predicted label (class with highest probability)
#         predictions = torch.argmax(logits, dim=-1)

#         # Convert probabilities to a list
#         predicted_probabilities = probabilities.numpy().tolist()[0]  # Convert from tensor to list

#         # Convert predictions to a list of emotion labels
#         predicted_labels = predictions.numpy().tolist()

#         # Map the numeric prediction to emotion label
#         predicted_label = dict1[predicted_labels[0]]

#         # Format probabilities as a string
#         prob_str = ', '.join([f"{dict1[i]}: {predicted_probabilities[i]:.4f}" for i in range(len(dict1))])

#         # Log or print the input, predicted label, and probabilities for debugging
#         print(f"Input: {input_text} | \nPredicted Label: {predicted_label} | Probabilities: {prob_str}")

#         # Return the prediction and probabilities in a JSON response
#         return jsonify({
#             'emotion_label': predicted_label,
#             'probabilities': {dict1[i]: f"{predicted_probabilities[i]:.4f}" for i in range(len(dict1))}
#         })

#     except Exception as e:
#         # Handle any errors and return an error message
#         return jsonify({'error': str(e)}), 500


from huggingface_hub import login
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.nn import functional as F 
# Replace with your Hugging Face token
token = "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"

# Log in using the Hugging Face token
login(token)
model_name = "jm12312/mttm_1" 
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)