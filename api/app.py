from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# Initialize FastAPI app
app = FastAPI()

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

# Updated category mapping to handle additional categories
category_mapping = {
    0: 'Financial Fraud', 
    1: 'Identity Theft',
    2: 'Phishing Scam',
    3: 'Ransomware Attack',
    4: 'Social Media Fraud',
    5: 'Cryptocurrency Scam',
    6: 'Ecommerce Fraud',
    7: 'Banking Fraud',
    8: 'Other' 
}

# Detect victim type and gender functions
def detect_victim_type(text):
    victim_keywords = {
        'individual': r'\b(I|me|my|individual|person)\b',
        'company': r'\b(company|business|organization|corporate)\b',
        'family_member': r'\b(mother|father|brother|family)\b',
        'friend': r'\b(friend|acquaintance)\b'
    }
    return [label for label, pattern in victim_keywords.items() if re.search(pattern, text.lower())] or ["unknown"]

def detect_gender(text):
    if re.search(r'\b(she|her)\b', text.lower()):
        return "female"
    elif re.search(r'\b(he|his)\b', text.lower()):
        return "male"
    return "unknown"

def extract_features(text):
    return {
        'contains_financial_terms': bool(re.search(r'bank|credit|money', text.lower())),
        'contains_urgency': bool(re.search(r'urgent|immediate|emergency', text.lower())),
        'contains_threats': bool(re.search(r'threat|hack|stolen', text.lower())),
        'message_length': len(text.split()),
        'has_numbers': bool(re.search(r'\d', text)),
    }

# Input data model
class TextInput(BaseModel):
    text: str

# Classification endpoint
@app.post("/classify")
async def classify_text(input: TextInput):
    text = input.text

    # Tokenize and classify
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
    top_category_idx = np.argmax(probs)
    top_category_label = category_mapping.get(top_category_idx, "Unknown")
    confidence_score = probs[top_category_idx]

    # Extract features
    extracted_features = extract_features(text)
    victim_type = detect_victim_type(text)
    gender = detect_gender(text)

    # Return JSON response
    return {
        "predicted_category": top_category_label,
        "confidence_score": round(float(confidence_score), 2),
        "all_scores": {category_mapping.get(i, "Unknown"): round(float(probs[i]), 2) for i in range(len(probs))},
        "features": extracted_features,
        "victim_type": victim_type,
        "gender": gender
    }

# To run the app, use the command:
# uvicorn app:app --reload
