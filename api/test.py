import torch
import numpy as np
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# Load the trained model and tokenizer
model_path = './trained_category_model'  # Adjust to your saved model path
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Ensure model is in evaluation mode and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Category mapping (use the same mapping as training)
category_mapping = {
    0: 'Financial Fraud',
    1: 'Identity Theft',
    2: 'Phishing Scam',
    3: 'Ransomware Attack',
    4: 'Social Media Fraud',
    5: 'Cryptocurrency Scam',
    6: 'Ecommerce Fraud',
    7: 'Banking Fraud'
}
# Reverse the category mapping
reverse_category_mapping = {v: k for k, v in category_mapping.items()}

# Define additional terms for feature extraction
cyber_terms = {
    'phishing': 'phish',
    'hacked': 'hack',
    'scammed': 'scam',
    'fraudulent': 'fraud'
}

features = {
    'contains_financial_terms': lambda text: bool(re.search(r'bank|credit|upi|money|payment', text.lower())),
    'contains_urgency': lambda text: bool(re.search(r'urgent|immediate|quick|emergency', text.lower())),
    'contains_threats': lambda text: bool(re.search(r'threat|hack|stolen|compromise', text.lower())),
    'message_length': lambda text: len(text.split()),
    'has_numbers': lambda text: bool(re.search(r'\d', text))
}

# Victim type detection function
def detect_victim_type(text):
    victim_keywords = {
        'individual': r'\b(I|me|my|myself|individual|person)\b',
        'company': r'\b(company|business|organization|firm|corporate)\b',
        'family_member': r'\b(mother|father|brother|sister|family|son|daughter)\b',
        'friend': r'\b(friend|acquaintance|buddy|pal)\b'
    }
    detected_types = [label for label, pattern in victim_keywords.items() if re.search(pattern, text.lower())]
    return detected_types or ["unknown"]

# Gender detection function
def detect_gender(text):
    if re.search(r'\b(she|her|hers)\b', text.lower()):
        return "female"
    elif re.search(r'\b(he|his|him)\b', text.lower()):
        return "male"
    else:
        return "unknown"

# Additional relationship and age detection function
def detect_relationship_context(text):
    relationships = {
        'child': r'\b(child|children|kid|son|daughter)\b',
        'teenager': r'\b(teen|teenager|young)\b',
        'adult': r'\b(adult|man|woman|lady|gentleman)\b',
        'elderly': r'\b(elderly|senior|old)\b',
    }
    detected_relationships = [label for label, pattern in relationships.items() if re.search(pattern, text.lower())]
    return detected_relationships or ["unknown"]

# Bigram and Trigram Analysis function
def extract_ngrams(text, n=2):
    words = text.lower().split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

# Function for entity recognition and analysis
def entity_recognition_analysis(text):
    entities = {}
    if re.search(r'\b[A-Z][a-z]+\b', text):  # Simple detection of names starting with uppercase
        entities['person_name'] = True
    if re.search(r'\bLtd|Inc|Corp|Company|Organization\b', text):
        entities['organization'] = True
    if re.search(r'\bStreet|Avenue|District|City\b', text):
        entities['location'] = True
    return entities

# Define a function to classify a single text, print confidence scores and features
def classify_text(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
    top_category_idx = np.argmax(probs)
    top_category_label = category_mapping[top_category_idx]
    confidence_score = probs[top_category_idx]

    # Extract features and print results
    extracted_features = {k: v(text) for k, v in features.items()}
    extracted_ngrams = extract_ngrams(text, n=2) + extract_ngrams(text, n=3)
    entity_info = entity_recognition_analysis(text)

    # Additional victim and relationship information
    victim_type = detect_victim_type(text)
    gender = detect_gender(text)
    relationship_context = detect_relationship_context(text)

    # Print the results
    print(f"Text: {text}")
    print(f"Predicted Category: {top_category_label}")
    print(f"Confidence Score: {confidence_score:.2f}")
    print(f"All Scores: {dict(zip(category_mapping.values(), probs))}")
    print("Extracted Features:", extracted_features)
    print("Extracted N-Grams:", extracted_ngrams)
    print("Entity Recognition Info:", entity_info)
    print("Victim Type:", victim_type)
    print("Detected Gender:", gender)
    print("Relationship Context:", relationship_context)
    print("\n")

# Sample texts for testing
sample_texts = [
    "Received a phishing email pretending to be my bank.",
    "Someone hacked into my social media account and posted inappropriate content.",
    "I lost money through an online cryptocurrency scam.",
    "A website I visited contained unlawful content and fake advertisements.",
    "I was a victim of identity theft through online impersonation.",
    "I never imagined something like this could happen to me, especially since Im so cautious. But then I started noticing that my online activity felt different, slower even. I cant even describe it well because everything seemed normal, except for the fact that I was constantly getting logged out of my accounts. I dont know how, but it’s like they knew everything I was doing, even though I was trying to be careful. It’s terrifying to think about. The more I tried to fix it, the more problems came up. My work account was notifiction, and then I got locked out of everything. I feel so helpless, like there’s nothing I can do to fix this. No matter how many times I change my password, it’s like someone’s always one step ahead. Even my laptop isn’t working right anymore. It’s so stange how everything just fell apart after that one email.",
    "time may be incoming call on pooja sharma asked job on bank varienti already search for job so i asked to job site send your whatsapp enter click the link regester your detailafter i enter link my mobile was hacked then enter the link enter our deatil make payment rs enter your debit card detail i submit my card detail after enter payment first amiunt debit rs i have doubt exit the webiste after automatcally enter otp debited my amount after my account balance rs only third person block whatsapp after minutes incoming call on IOB chennai asked your debit oon any scam i to told further block my debit card on IOB bank side thank you",
    "Respected sir some unknown culprit is cyber bullying me with mail id smggmailcom since with false allegations to our district collector that I am not caring my higher officers allegation of avoiding of holding additional duties allegation of my income sources etc and also continuously cyberbullying me nearly times so please take necessary action on my petition as soon as possible",
    "While playing online game email id and password he has shared and email was hacked Email id that has been hacked is puneeshagarwalgmailcom I am not able to login now All my contacts data and emails are in this account"
]

# Run classification on sample texts
for text in sample_texts:
    classify_text(text)
