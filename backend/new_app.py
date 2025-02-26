# from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
# import torch
# # Replace with your model's path (or username/model-name)
# from huggingface_hub import login

# # Replace with your Hugging Face token
# token = "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"

# # Log in using the Hugging Face token
# login(token)

# model_name = "jm12312/mttm_1" 

# # Load the model from Hugging Face
# model = DistilBertForSequenceClassification.from_pretrained(model_name)

# # Load the tokenizer
# tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# text = "jews are very responsible for all the problems in this country.".lower()
# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# # Set the model to evaluation mode
# model.eval()

# # Make predictions
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits

# # Get the predicted class (for classification)
# predicted_class = torch.argmax(logits, dim=-1).item()
# print(predicted_class)