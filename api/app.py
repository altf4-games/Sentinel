from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import re
import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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

# Helper functions
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

    # Softmax to get probabilities
    probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
    top_category_idx = np.argmax(probs)
    
    # Reverse mapping to find category name by index
    top_category_label = {v: k for k, v in category_mapping.items()}.get(top_category_idx, "Unknown")
    confidence_score = probs[top_category_idx]

    # Debugging output to ensure correct mappings
    print("Logits:", logits)
    print("Probabilities:", probs)
    print("Top category index:", top_category_idx)
    print("Predicted category:", top_category_label)
    print("Confidence score:", confidence_score)

    # Extract features
    extracted_features = extract_features(text)
    victim_type = detect_victim_type(text)
    gender = detect_gender(text)

    # Prepare JSON response
    return {
        "predicted_category": top_category_label,
        "confidence_score": round(float(confidence_score), 2),
        "all_scores": {cat: round(float(probs[idx]), 2) for cat, idx in category_mapping.items()},
        "features": extracted_features,
        "victim_type": victim_type,
        "gender": gender
    }

# To run the app, use the command:
# uvicorn app:app --reload
