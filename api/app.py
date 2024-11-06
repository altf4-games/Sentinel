from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import torch
import numpy as np
import re
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
import torch.nn.functional as F

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
model_path = './trained_category_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load category mapping from JSON
with open(f"{model_path}/category_mapping.json", "r") as f:
    category_mapping = json.load(f)

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Helper functions
def detect_victim_type(text):
    victim_keywords = {
        'individual': r'\b(I|me|my|individual|person)\b',
        'company': r'\b(company|business|organization|corporate)\b',
        'family_member': r'\b(mother|father|brother|family)\b',
        'friend': r'\b(friend|acquaintance)\b'
    }
    return [label for label, pattern in victim_keywords.items() if re.search(pattern, text.lower())] or ["unknown"]

def extract_sensitive_data(text):
    # Refined patterns for sensitive information
    patterns = {
        "upi_ids": r'\b[\w.-]+@[a-zA-Z]+\b',  # UPI ID typically has a format like "username@bankname"
        "emails": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Standard email format
        "phone_numbers": r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b',  # Indian phone numbers with optional country code
        "websites": r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.(?:com|org|net|in|edu|gov|co|info|biz|me|io|tech)\b',  # Common website formats
        "other_numbers": r'\b\d{6,16}\b'  # Other numbers, such as IDs or account numbers, between 6 to 16 digits
    }
    
    # Using regex to find all matches
    sensitive_data = {key: re.findall(pattern, text) for key, pattern in patterns.items()}
    return sensitive_data

def extract_features(text):
    return {
        'contains_financial_terms': bool(re.search(r'bank|credit|money', text.lower())),
        'contains_urgency': bool(re.search(r'urgent|immediate|emergency', text.lower())),
        'contains_threats': bool(re.search(r'threat|hack|stolen', text.lower())),
        'message_length': len(text.split()),
        'has_numbers': bool(re.search(r'\d', text)),
    }

# Helper function for text classification
async def classify_text(text: str):
    # Tokenize and classify
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Softmax to get probabilities
    probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
    top_category_idx = np.argmax(probs)
    top_category_label = {v: k for k, v in category_mapping.items()}.get(top_category_idx, "Unknown")
    confidence_score = probs[top_category_idx]

    # Sentiment analysis
    sentiment_result = sentiment_analyzer(text)
    sentiment_label = sentiment_result[0]["label"]
    sentiment_score = sentiment_result[0]["score"]

    # Extract features, victim type, sensitive data, etc.
    extracted_features = extract_features(text)
    victim_type = detect_victim_type(text)
    sensitive_data = extract_sensitive_data(text)

    return {
        "predicted_category": top_category_label,
        "confidence_score": round(float(confidence_score), 2),
        "all_scores": {cat: round(float(probs[idx]), 2) for cat, idx in category_mapping.items()},
        "features": extracted_features,
        "victim_type": victim_type,
        "sensitive_data": sensitive_data,
        "sentiment": {
            "label": sentiment_label,
            "score": round(sentiment_score, 2)
        }
    }

# Endpoint to classify plain text
class TextInput(BaseModel):
    text: str

@app.post("/classify")
async def classify(input: TextInput):
    return await classify_text(input.text)

# OCR endpoint that calls /classify after extracting text
@app.post("/ocr")
async def ocr_and_classify(file: UploadFile = File(...)):
    ocr_api_url = "https://api.ocr.space/parse/image"
    payload = {
        'apikey': 'K83989354488957', # The API key is provided for easier testing, will be removed in production
        'isOverlayRequired': False,
        'language': 'eng'
    }
    files = {'file': (file.filename, await file.read(), file.content_type)}
    
    # Call the OCR API
    response = requests.post(ocr_api_url, files=files, data=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="OCR failed")
    
    # Parse OCR result
    result = response.json()
    if result.get("IsErroredOnProcessing", True):
        raise HTTPException(status_code=500, detail="Error processing image")

    ocr_text = result.get("ParsedResults")[0].get("ParsedText")
    if not ocr_text:
        raise HTTPException(status_code=400, detail="No text extracted from image")

    # Pass extracted text to classify function
    classification_result = await classify_text(ocr_text)
    return classification_result
