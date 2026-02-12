# Configuration file for the pipeline

import os
from dotenv import load_dotenv


load_dotenv()

CONFIG = {
    # NER model settings
    "ner": {
        "model_name": "d4data/biomedical-ner-all",
        "confidence_threshold": 0.8,
        "batch_size": 8,
        "device": "auto"
    },
    
    # Sentiment model settings (Transformer-based, task compliant)
    "sentiment": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "device": "auto",
        "positive_threshold": 0.65,
        "negative_threshold": 0.65
    },
    
    # Gemini API settings
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model_name": "gemini-2.5-flash",
        "temperature": 0.0,
        "max_tokens": 2048,
        "top_p": 0.95,
        "top_k": 40,
        "safety_settings": {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
    },
    
    # Speaker patterns
    "speakers": {
        "patterns": {
            "doctor": [
                r'(doctor|physician|dr\.?)\s*:',
                r'(provider|clinician|practitioner)\s*:',
                r'MD\s*:',
            ],
            "patient": [
                r'(patient|pt\.?)\s*:',
                r'(client|individual)\s*:',
            ]
        },

        "canonical_names": {
            "doctor": "doctor",
            "patient": "patient"
        }
    },
    
    # Entity categories
    "entity_categories": {

        "symptoms": ["SYMPTOM", "SIGN", "FINDING"],
        "diseases": ["DISEASE", "CONDITION", "DISORDER", "SYNDROME"],
        "treatments": ["TREATMENT", "PROCEDURE", "THERAPY", "INTERVENTION"],
        "anatomy": ["ANATOMY", "BODY_PART", "ORGAN", "TISSUE", "ANATOMICAL"],
        "medications": ["MEDICATION", "DRUG", "PHARMACEUTICAL", "MEDICINE"],
    },
    "entity_default_category": "other",
    
    # Text limits
    "text_limits": {
        "ner_max_chars": 5000,
        "llm_max_chars": 4000,
        "top_entities_for_llm": 25,
        "min_entity_length": 2,
    },
    
    # Output settings
    "output": {
        "save_intermediate": True,
        "pretty_print": True,
        "include_metadata": True,
        "filenames": {
            "summary": "medical_summary.json",
            "sentiment": "sentiment_intent.json",
            "soap": "soap_note.json",
            "entities": "entities.json",
            "complete": "complete_results.json"
        }
    },
    
    # Preprocessing options
    "preprocessing": {
        "filler_words": ['um', 'uh', 'like', 'you know', 'i mean', 'well'],
        "remove_fillers": False,
        "normalize_whitespace": True,
    },
    
    # Sentiment and intent options
    "sentiment_classes": ["Anxious", "Neutral", "Reassured"],
    
    "intent_labels": [
        "Reporting symptoms",
        "Seeking reassurance",
        "Expressing improvement",
        "Asking questions",
        "Neutral update"
    ]
}

# Validate API key
if not CONFIG["gemini"]["api_key"]:
    raise ValueError(
        "❌ GEMINI_API_KEY not found!\n"
        "Please set it in .env file.\n"
        "Get your key from: https://makersuite.google.com/app/apikey"
    )


SENTIMENT_CLASSES = CONFIG["sentiment_classes"]
INTENT_LABELS = CONFIG["intent_labels"]
